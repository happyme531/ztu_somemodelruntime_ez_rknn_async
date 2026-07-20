#include <ztu/somemodelruntime_rknn.hpp>

#include "ez_rknn_async.hpp"
#include "ez_rknn_zero_copy.hpp"
#include "rknpu3_backend.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <mutex>
#include <queue>
#include <set>
#include <sstream>
#include <utility>

namespace ztu::somemodelruntime::rknn {
namespace {

using ztu::rk::AsyncEzRknn;
using ztu::rk::RknnContextHolder;
using ztu::rk::RknnIoBinding;
using ztu::rk::RknnOrtValue;
using ztu::rk::Rknpu3IoBinding;
using ztu::rk::Rknpu3Session;
using ztu::rk::ZeroCopyEzRknn;
using ztu::rk::ZeroCopyInputLayout;
using ztu::rk::ZeroCopyLayoutMode;
using ztu::rk::attr_shape_i64;
using ztu::rk::require_shape_matches;
using ztu::rk::rknn_tensor_type_size;
using ztu::rk::shape_to_string;

ExecutionProvider detect_model_provider(const void *data, size_t size) {
  if (data == nullptr || size < 4)
    throw Exception("RKNN model is too small to contain a valid header");
  const auto *magic = static_cast<const uint8_t *>(data);
  if (std::memcmp(magic, "RKNN", 4) == 0)
    return ExecutionProvider::Rknpu2;
  if (std::memcmp(magic, "RKNX", 4) == 0)
    return ExecutionProvider::Rknpu3;
  std::ostringstream text;
  text << "Unknown RKNN model magic: ";
  for (size_t i = 0; i < 4; ++i) {
    if (i != 0)
      text << ' ';
    text << std::hex << static_cast<unsigned>(magic[i]);
  }
  throw Exception(text.str());
}

ExecutionProvider detect_model_provider(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    throw Exception("Unable to open RKNN model: " + path.string());
  std::array<uint8_t, 4> magic = {};
  stream.read(reinterpret_cast<char *>(magic.data()), magic.size());
  if (stream.gcount() != static_cast<std::streamsize>(magic.size()))
    throw Exception("RKNN model is too small: " + path.string());
  return detect_model_provider(magic.data(), magic.size());
}

ExecutionProvider select_model_provider(ExecutionProvider requested,
                                        ExecutionProvider detected) {
  if (requested == ExecutionProvider::Auto)
    return detected;
  if (requested != detected) {
    const char *requested_name =
        requested == ExecutionProvider::Rknpu2 ? "RKNPU2" : "RKNPU3";
    const char *detected_name =
        detected == ExecutionProvider::Rknpu2 ? "RKNPU2" : "RKNPU3";
    throw Exception(std::string("Provider/model mismatch: requested ") +
                    requested_name + " but the model header is " +
                    detected_name);
  }
  return requested;
}

rknn_tensor_type to_rknn_type(TensorElementType type) {
  switch (type) {
  case TensorElementType::Float32:
    return RKNN_TENSOR_FLOAT32;
  case TensorElementType::Float16:
    return RKNN_TENSOR_FLOAT16;
  case TensorElementType::Int8:
    return RKNN_TENSOR_INT8;
  case TensorElementType::Uint8:
    return RKNN_TENSOR_UINT8;
  case TensorElementType::Int16:
    return RKNN_TENSOR_INT16;
  case TensorElementType::Uint16:
    return RKNN_TENSOR_UINT16;
  case TensorElementType::Int32:
    return RKNN_TENSOR_INT32;
  case TensorElementType::Uint32:
    return RKNN_TENSOR_UINT32;
  case TensorElementType::Int64:
    return RKNN_TENSOR_INT64;
  case TensorElementType::Bool:
    return RKNN_TENSOR_BOOL;
  default:
    return RKNN_TENSOR_TYPE_MAX;
  }
}

TensorElementType from_rknn_type(rknn_tensor_type type) {
  switch (type) {
  case RKNN_TENSOR_FLOAT32:
    return TensorElementType::Float32;
  case RKNN_TENSOR_FLOAT16:
    return TensorElementType::Float16;
  case RKNN_TENSOR_INT8:
    return TensorElementType::Int8;
  case RKNN_TENSOR_UINT8:
    return TensorElementType::Uint8;
  case RKNN_TENSOR_INT16:
    return TensorElementType::Int16;
  case RKNN_TENSOR_UINT16:
    return TensorElementType::Uint16;
  case RKNN_TENSOR_INT32:
    return TensorElementType::Int32;
  case RKNN_TENSOR_UINT32:
    return TensorElementType::Uint32;
  case RKNN_TENSOR_INT64:
    return TensorElementType::Int64;
  case RKNN_TENSOR_BOOL:
    return TensorElementType::Bool;
  default:
    return TensorElementType::Unknown;
  }
}

std::string onnx_type_string(rknn_tensor_type type) {
  switch (type) {
  case RKNN_TENSOR_FLOAT32:
    return "tensor(float)";
  case RKNN_TENSOR_FLOAT16:
    return "tensor(float16)";
  case RKNN_TENSOR_INT8:
    return "tensor(int8)";
  case RKNN_TENSOR_UINT8:
    return "tensor(uint8)";
  case RKNN_TENSOR_INT16:
    return "tensor(int16)";
  case RKNN_TENSOR_UINT16:
    return "tensor(uint16)";
  case RKNN_TENSOR_INT32:
    return "tensor(int32)";
  case RKNN_TENSOR_UINT32:
    return "tensor(uint32)";
  case RKNN_TENSOR_INT64:
    return "tensor(int64)";
  case RKNN_TENSOR_BOOL:
    return "tensor(bool)";
  default:
    return "tensor(unknown)";
  }
}

size_t checked_element_count(const std::vector<int64_t> &shape) {
  size_t result = 1;
  for (int64_t dim : shape) {
    if (dim < 0) {
      throw Exception("shape dims must be >= 0");
    }
    if (dim != 0 && result > std::numeric_limits<size_t>::max() /
                                 static_cast<size_t>(dim)) {
      throw Exception("tensor shape is too large");
    }
    result *= static_cast<size_t>(dim);
  }
  return result;
}

rknn_core_mask to_core_mask(CoreMask mask) {
  switch (mask) {
  case CoreMask::Auto:
    return RKNN_NPU_CORE_AUTO;
  case CoreMask::All:
    return RKNN_NPU_CORE_ALL;
  case CoreMask::Core0:
    return RKNN_NPU_CORE_0;
  case CoreMask::Core1:
    return RKNN_NPU_CORE_1;
  case CoreMask::Core2:
    return RKNN_NPU_CORE_2;
  case CoreMask::Core01:
    return RKNN_NPU_CORE_0_1;
  case CoreMask::Core012:
    return RKNN_NPU_CORE_0_1_2;
  }
  return RKNN_NPU_CORE_AUTO;
}

AsyncEzRknn::Layout to_async_layout(Layout layout, bool &software_nchw) {
  software_nchw = false;
  switch (layout) {
  case Layout::Nchw:
    return AsyncEzRknn::Layout::NCHW;
  case Layout::NchwSoftware:
    software_nchw = true;
    return AsyncEzRknn::Layout::NHWC;
  case Layout::Nhwc:
    return AsyncEzRknn::Layout::NHWC;
  case Layout::Any:
    return AsyncEzRknn::Layout::ANY;
  }
  return AsyncEzRknn::Layout::NCHW;
}

ZeroCopyInputLayout to_zero_copy_input_layout(Layout layout) {
  if (layout == Layout::Nhwc)
    return ZeroCopyInputLayout::NHWC;
  if (layout == Layout::Any)
    return ZeroCopyInputLayout::ANY;
  return ZeroCopyInputLayout::NCHW;
}

ZeroCopyLayoutMode to_zero_copy_layout(ZeroCopyLayout layout) {
  return layout == ZeroCopyLayout::Native ? ZeroCopyLayoutMode::NATIVE
                                           : ZeroCopyLayoutMode::NHWC;
}

std::vector<NodeArg>
build_node_args(const std::vector<rknn_tensor_attr> &attrs,
                const std::vector<std::string> &names) {
  std::vector<NodeArg> result;
  result.reserve(attrs.size());
  for (size_t i = 0; i < attrs.size(); ++i) {
    std::string layout = "UNDEFINED";
    if (attrs[i].fmt == RKNN_TENSOR_NCHW)
      layout = "NCHW";
    else if (attrs[i].fmt == RKNN_TENSOR_NHWC)
      layout = "NHWC";
    else if (attrs[i].fmt == RKNN_TENSOR_NC1HWC2)
      layout = "NC1HWC2";
    result.push_back(NodeArg{names.at(i), attr_shape_i64(attrs[i]),
                             onnx_type_string(attrs[i].type),
                             std::move(layout), {}});
  }
  return result;
}

std::vector<NodeArg> build_rknpu3_node_args(
    const std::vector<ztu::rk::rknpu3_abi::TensorAttr> &attrs,
    const std::vector<std::string> &names) {
  std::vector<NodeArg> result;
  result.reserve(attrs.size());
  for (size_t i = 0; i < attrs.size(); ++i) {
    const auto type = ztu::rk::rknpu3_attr_rknpu2_compatible_type(attrs[i]);
    result.push_back(NodeArg{
        names.at(i), ztu::rk::rknpu3_attr_shape(attrs[i]),
        onnx_type_string(type), ztu::rk::rknpu3_layout_name(attrs[i].layout),
        ztu::rk::rknpu3_attr_strides(attrs[i])});
  }
  return result;
}

bool parse_bool_string(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](char ch) {
    return static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  });
  if (value == "1" || value == "true" || value == "on" || value == "yes")
    return true;
  if (value == "0" || value == "false" || value == "off" || value == "no")
    return false;
  throw Exception("run config value must be a boolean");
}

} // namespace

struct Env::Impl {
  std::string log_id;
  WarningHandler warning_handler;
};

Env::Env(std::string log_id) : impl_(std::make_shared<Impl>()) {
  impl_->log_id = std::move(log_id);
  impl_->warning_handler = [](std::string_view message) {
    std::cerr << "[ztu_somemodelruntime_rknn] warning: " << message
              << std::endl;
  };
}

void Env::SetWarningHandler(WarningHandler handler) {
  impl_->warning_handler = std::move(handler);
}

SessionOptions::SessionOptions() = default;

SessionOptions &
SessionOptions::SetExecutionProvider(ExecutionProvider provider) {
  provider_ = provider;
  return *this;
}

ExecutionProvider SessionOptions::GetExecutionProvider() const {
  return provider_;
}

SessionOptions &
SessionOptions::SetRknnProviderOptions(const RknnProviderOptions &options) {
  options_ = options;
  return *this;
}

const RknnProviderOptions &SessionOptions::GetRknnProviderOptions() const {
  return options_;
}

SessionOptions &SessionOptions::SetRknpu3ProviderOptions(
    const Rknpu3ProviderOptions &options) {
  rknpu3_options_ = options;
  return *this;
}

const Rknpu3ProviderOptions &
SessionOptions::GetRknpu3ProviderOptions() const {
  return rknpu3_options_;
}

RunOptions &RunOptions::AddConfigEntry(std::string key, std::string value) {
  entries_[std::move(key)] = std::move(value);
  return *this;
}

RunOptions &RunOptions::SetDispatchBatch(bool enabled) {
  entries_["ztu_modelrt_dispatch_batch"] = enabled ? "1" : "0";
  return *this;
}

bool RunOptions::GetDispatchBatch() const {
  auto it = entries_.find("ztu_modelrt_dispatch_batch");
  return it != entries_.end() && parse_bool_string(it->second);
}

MemoryInfo MemoryInfo::CreateCpu(int device_id) {
  return MemoryInfo(DeviceType::Cpu, device_id);
}

MemoryInfo MemoryInfo::CreateRknpu2(int device_id) {
  return MemoryInfo(DeviceType::Rknpu2, device_id);
}

MemoryInfo MemoryInfo::CreateRknpu3(int device_id) {
  return MemoryInfo(DeviceType::Rknpu3, device_id);
}

const char *MemoryInfo::GetName() const noexcept {
  switch (device_type_) {
  case DeviceType::Cpu:
    return "cpu";
  case DeviceType::Rknpu2:
    return "rknpu2";
  case DeviceType::Rknpu3:
    return "rknpu3";
  }
  return "unknown";
}

TensorTypeAndShapeInfo::TensorTypeAndShapeInfo(TensorElementType type,
                                               std::vector<int64_t> shape)
    : type_(type), shape_(std::move(shape)) {}

size_t TensorTypeAndShapeInfo::GetElementCount() const {
  return checked_element_count(shape_);
}

TypeInfo::TypeInfo(TensorTypeAndShapeInfo tensor_info)
    : tensor_info_(std::move(tensor_info)) {}

const TensorTypeAndShapeInfo &
TypeInfo::GetTensorTypeAndShapeInfo() const noexcept {
  return tensor_info_;
}

const char *TensorElementTypeName(TensorElementType type) noexcept {
  switch (type) {
  case TensorElementType::Float32:
    return "float32";
  case TensorElementType::Float16:
    return "float16";
  case TensorElementType::Int8:
    return "int8";
  case TensorElementType::Uint8:
    return "uint8";
  case TensorElementType::Int16:
    return "int16";
  case TensorElementType::Uint16:
    return "uint16";
  case TensorElementType::Int32:
    return "int32";
  case TensorElementType::Uint32:
    return "uint32";
  case TensorElementType::Int64:
    return "int64";
  case TensorElementType::Bool:
    return "bool";
  default:
    return "unknown";
  }
}

Value Value::CreateTensorImpl(const MemoryInfo &memory_info, void *data,
                              size_t bytes, const int64_t *shape,
                              size_t shape_len, TensorElementType type) {
  if (shape == nullptr && shape_len != 0)
    throw Exception("shape is null");
  std::vector<int64_t> dims(shape, shape + shape_len);
  const rknn_tensor_type native_type = to_rknn_type(type);
  if (native_type == RKNN_TENSOR_TYPE_MAX)
    throw Exception("unsupported tensor element type");
  const size_t expected =
      checked_element_count(dims) * rknn_tensor_type_size(native_type);
  if (bytes != expected)
    throw Exception("tensor buffer size does not match shape and type");
  if (memory_info.GetDeviceType() == DeviceType::Cpu) {
    return Value(RknnOrtValue::from_cpu_buffer(data, bytes, std::move(dims),
                                               native_type));
  }
  if (memory_info.GetDeviceType() == DeviceType::Rknpu3) {
    throw Exception("Standalone RKNPU3 tensors require a session/name/io_kind; "
                    "use Session::CreateIoValue");
  }
  auto value = RknnOrtValue::create_generic_rknn(std::move(dims), native_type,
                                                  0);
  value->update_inplace(data, bytes, value->shape(), native_type);
  return Value(std::move(value));
}

Value Value::CreateTensor(const MemoryInfo &memory_info, void *data,
                          size_t bytes, const int64_t *shape, size_t shape_len,
                          TensorElementType type) {
  return CreateTensorImpl(memory_info, data, bytes, shape, shape_len, type);
}

Value Value::CreateOwnedTensor(const int64_t *shape, size_t shape_len,
                               TensorElementType type) {
  if (shape == nullptr && shape_len != 0)
    throw Exception("shape is null");
  const rknn_tensor_type native_type = to_rknn_type(type);
  if (native_type == RKNN_TENSOR_TYPE_MAX)
    throw Exception("unsupported tensor element type");
  return Value(RknnOrtValue::allocate_cpu(
      std::vector<int64_t>(shape, shape + shape_len), native_type));
}

bool Value::IsTensor() const noexcept { return impl_ && impl_->is_tensor(); }
bool Value::HasValue() const noexcept { return impl_ && impl_->has_value(); }

void *Value::GetTensorMutableRawData() {
  if (!impl_)
    throw Exception("Value is empty");
  return reinterpret_cast<void *>(impl_->data_ptr());
}

const void *Value::GetTensorRawData() const {
  if (!impl_)
    throw Exception("Value is empty");
  return reinterpret_cast<const void *>(impl_->data_ptr());
}

size_t Value::GetTensorSizeInBytes() const {
  if (!impl_)
    throw Exception("Value is empty");
  return impl_->logical_bytes();
}

TensorTypeAndShapeInfo Value::GetTensorTypeAndShapeInfo() const {
  if (!impl_)
    throw Exception("Value is empty");
  return TensorTypeAndShapeInfo(from_rknn_type(impl_->element_type()),
                                impl_->shape());
}

ValueMemoryInfo Value::GetMemoryInfo() const {
  if (!impl_)
    throw Exception("Value is empty");
  const auto info = impl_->memory_info();
  ValueMemoryInfo result;
  if (info.device_type == "rknpu2")
    result.device_type = DeviceType::Rknpu2;
  else if (info.device_type == "rknpu3")
    result.device_type = DeviceType::Rknpu3;
  else
    result.device_type = DeviceType::Cpu;
  result.device_id = info.device_id;
  result.size = info.size;
  result.fd = info.fd;
  result.offset = info.offset;
  result.virt_addr = info.virt_addr;
  result.import_count = info.import_count;
  result.dmabuf_backed = info.dmabuf_backed;
  result.attr_size = info.attr_size;
  result.size_with_stride = info.size_with_stride;
  result.w_stride = info.w_stride;
  result.h_stride = info.h_stride;
  result.fmt = info.fmt;
  result.pass_through = info.pass_through;
  result.aligned_size = info.aligned_size;
  result.core_id = info.core_id;
  result.layout = info.layout;
  result.strides = info.strides;
  return result;
}

void Value::Update(const void *data, size_t bytes,
                   const std::vector<int64_t> &shape,
                   TensorElementType type) {
  if (!impl_)
    throw Exception("Value is empty");
  impl_->update_inplace(data, bytes, shape, to_rknn_type(type));
}

void Value::CopyTo(void *data, size_t bytes) const {
  if (!impl_)
    throw Exception("Value is empty");
  impl_->copy_to(data, bytes);
}

void Value::SyncToDevice() {
  if (!impl_)
    throw Exception("Value is empty");
  impl_->sync_to_device();
}

void Value::SyncFromDevice() {
  if (!impl_)
    throw Exception("Value is empty");
  impl_->sync_from_device();
}

struct ValueAccess {
  static const std::shared_ptr<RknnOrtValue> &get(const Value &value) {
    return value.impl_;
  }
  static Value make(std::shared_ptr<RknnOrtValue> impl) {
    return Value(std::move(impl));
  }
};

struct Session::Impl : std::enable_shared_from_this<Session::Impl> {
  struct PreparedInput {
    std::shared_ptr<RknnOrtValue> owner;
    std::vector<uint8_t> transformed;
    AsyncEzRknn::InputView view;
  };

  using ResultPack =
      std::pair<AsyncEzRknn::InferenceResponse, std::vector<rknn_tensor_attr>>;

  std::shared_ptr<Env::Impl> env;
  ExecutionProvider provider = ExecutionProvider::Auto;
  RknnProviderOptions options;
  Rknpu3ProviderOptions rknpu3_options;
  std::shared_ptr<AsyncEzRknn> async;
  std::shared_ptr<ZeroCopyEzRknn> zero_copy;
  std::shared_ptr<Rknpu3Session> rknpu3;
  std::shared_ptr<Impl> weight_master;
  std::vector<rknn_tensor_attr> input_attrs;
  std::vector<rknn_tensor_attr> output_attrs;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  bool nchw_software = false;
  std::chrono::milliseconds submit_timeout{10000};
  size_t pipeline_depth = 0;
  bool pipeline_initialized = false;
  std::queue<std::future<ResultPack>> pipeline_queue;
  std::mutex pipeline_mutex;
  std::mutex async_callback_mutex;
  std::condition_variable async_callback_cv;
  size_t pending_async_callbacks = 0;

  void begin_async_callback() {
    std::lock_guard<std::mutex> lock(async_callback_mutex);
    ++pending_async_callbacks;
  }

  void end_async_callback() {
    {
      std::lock_guard<std::mutex> lock(async_callback_mutex);
      --pending_async_callbacks;
    }
    async_callback_cv.notify_all();
  }

  void wait_for_async_callbacks() {
    std::unique_lock<std::mutex> lock(async_callback_mutex);
    async_callback_cv.wait(lock,
                           [this] { return pending_async_callbacks == 0; });
  }

  template <typename Factory>
  void initialize_rknpu2(Factory &&factory,
                         const SessionOptions &session_options,
                         const std::shared_ptr<Impl> &master) {
    if (master && master->provider != ExecutionProvider::Rknpu2)
      throw Exception("use_weight_from must refer to an RKNPU2 session");
    provider = ExecutionProvider::Rknpu2;
    options = session_options.GetRknnProviderOptions();
    if (options.max_queue_size == 0)
      throw Exception("max_queue_size must be > 0");
    if (options.threads_per_core <= 0)
      throw Exception("threads_per_core must be > 0");
    if (options.submit_timeout_ms <= 0)
      throw Exception("submit_timeout_ms must be > 0");
    if (options.schedule && options.schedule->empty())
      throw Exception("schedule must not be empty");
    if (options.schedule && options.tp_mode)
      throw Exception("tp_mode conflicts with schedule; set only one");

    const bool custom_ops = !options.custom_op_paths.empty() ||
                            options.custom_op_default_path;
    if (master && custom_ops)
      throw Exception("use_weight_from cannot be combined with custom ops");
    weight_master = master;
    const bool disable_dup = options.disable_dup_context || custom_ops || master;
    if (custom_ops)
      warn("Custom op loading requested; dup_context is disabled for this "
           "session to avoid known RKNN stability issues.");

    const auto layout = to_async_layout(options.layout, nchw_software);
    std::optional<rknn_context> master_context;
    if (master) {
      if (!master->async || master->async->contexts.empty())
        throw Exception("use_weight_from must be an initialized session");
      master_context = master->async->contexts.front();
    }
    const auto mask = options.tp_mode
                          ? std::optional<rknn_core_mask>(
                                to_core_mask(*options.tp_mode))
                          : std::optional<rknn_core_mask>(RKNN_NPU_CORE_AUTO);
    auto warning = [env = env](const std::string &message) {
      if (env && env->warning_handler)
        env->warning_handler(message);
    };
    async = factory(layout, mask, disable_dup, master_context, warning);
    if (async->sdk_version_warning())
      warn(*async->sdk_version_warning());
    for (const auto &path : options.custom_op_paths)
      async->load_custom_op(path);
    if (options.custom_op_default_path)
      async->load_custom_ops_from_default_path();
    if (custom_ops)
      async->register_loaded_custom_ops();
    if (async->contexts.empty() || async->contexts.front() == 0)
      throw Exception("Async RKNN context is not available");

    auto holder = std::make_shared<RknnContextHolder>(
        async->contexts.front(), false, std::static_pointer_cast<void>(async),
        async->context_mutex(0));
    zero_copy = std::make_shared<ZeroCopyEzRknn>(
        std::move(holder), to_zero_copy_input_layout(options.layout),
        to_zero_copy_layout(options.zero_copy_layout));
    input_attrs = async->input_attrs;
    output_attrs = async->output_attrs;
    if (nchw_software) {
      for (auto &attr : input_attrs) {
        if (attr.n_dims == 4) {
          const auto n = attr.dims[0];
          const auto h = attr.dims[1];
          const auto w = attr.dims[2];
          const auto c = attr.dims[3];
          attr.dims[0] = n;
          attr.dims[1] = c;
          attr.dims[2] = h;
          attr.dims[3] = w;
          if (attr.fmt == RKNN_TENSOR_NHWC)
            attr.fmt = RKNN_TENSOR_NCHW;
        }
      }
    }
    for (size_t i = 0; i < input_attrs.size(); ++i)
      input_names.push_back(input_attrs[i].name[0]
                                ? std::string(input_attrs[i].name)
                                : "input_" + std::to_string(i));
    for (size_t i = 0; i < output_attrs.size(); ++i)
      output_names.push_back(output_attrs[i].name[0]
                                 ? std::string(output_attrs[i].name)
                                 : "output_" + std::to_string(i));
    submit_timeout = std::chrono::milliseconds(options.submit_timeout_ms);
  }

  void initialize_rknpu3(const SessionOptions &session_options,
                         const std::string *model_path,
                         std::vector<uint8_t> model_data = {}) {
    provider = ExecutionProvider::Rknpu3;
    rknpu3_options = session_options.GetRknpu3ProviderOptions();
    auto warning = [env = env](const std::string &message) {
      if (env && env->warning_handler)
        env->warning_handler(message);
    };
    if (model_path != nullptr)
      rknpu3 = std::make_shared<Rknpu3Session>(*model_path, rknpu3_options,
                                               std::move(warning));
    else
      rknpu3 = std::make_shared<Rknpu3Session>(
          std::move(model_data), rknpu3_options, std::move(warning));
    input_names = rknpu3->input_names();
    output_names = rknpu3->output_names();
  }

  void require_rknpu2(const char *operation) const {
    if (provider != ExecutionProvider::Rknpu2)
      throw Exception(std::string(operation) +
                      " is not implemented by the initial RKNPU3 provider; "
                      "use the single-thread zero-copy IoBinding API");
  }

  void warn(const std::string &message) const {
    if (env && env->warning_handler)
      env->warning_handler(message);
  }

  size_t input_index(std::string_view name) const {
    auto it = std::find(input_names.begin(), input_names.end(), name);
    if (it == input_names.end())
      throw Exception("Unknown input name: " + std::string(name));
    return static_cast<size_t>(std::distance(input_names.begin(), it));
  }

  std::vector<size_t> output_indices(const char *const *names,
                                     size_t count) const {
    if (names == nullptr && count == 0) {
      std::vector<size_t> all(output_names.size());
      for (size_t i = 0; i < all.size(); ++i)
        all[i] = i;
      return all;
    }
    if (names == nullptr)
      throw Exception("output_names is null");
    std::vector<size_t> result;
    result.reserve(count);
    for (size_t n = 0; n < count; ++n) {
      auto it = std::find(output_names.begin(), output_names.end(), names[n]);
      if (it == output_names.end())
        throw Exception("Unknown output name: " + std::string(names[n]));
      result.push_back(
          static_cast<size_t>(std::distance(output_names.begin(), it)));
    }
    return result;
  }

  PreparedInput prepare_one(const Value &value, size_t index,
                            bool allow_batch) const {
    const auto &value_impl = ValueAccess::get(value);
    if (!value_impl || !value_impl->is_cpu())
      throw Exception("regular Run accepts CPU tensor Values; use IoBinding "
                      "for RKNPU2 Values");
    const auto &shape = value_impl->shape();
    const auto expected = attr_shape_i64(input_attrs.at(index));
    if (shape.size() != expected.size())
      throw Exception("Input rank mismatch for '" + input_names[index] + "'");
    if (allow_batch) {
      if (shape.empty() || shape[0] <= 0)
        throw Exception("Batch input must have a positive first dimension");
      for (size_t d = 1; d < shape.size(); ++d) {
        if (shape[d] != expected[d])
          throw Exception("Batch input '" + input_names[index] +
                          "' shape mismatch");
      }
    } else {
      require_shape_matches(shape, expected,
                            "Input '" + input_names[index] + "'");
    }

    PreparedInput result;
    result.owner = value_impl;
    const auto type = value_impl->element_type();
    const void *input_data =
        reinterpret_cast<const void *>(value_impl->data_ptr());
    std::vector<int64_t> native_shape = shape;
    if (nchw_software && shape.size() == 4) {
      const size_t n = static_cast<size_t>(shape[0]);
      const size_t c = static_cast<size_t>(shape[1]);
      const size_t h = static_cast<size_t>(shape[2]);
      const size_t w = static_cast<size_t>(shape[3]);
      const size_t elem = rknn_tensor_type_size(type);
      result.transformed.resize(n * c * h * w * elem);
      const auto *src = static_cast<const uint8_t *>(input_data);
      auto *dst = result.transformed.data();
      for (size_t ni = 0; ni < n; ++ni)
        for (size_t ci = 0; ci < c; ++ci)
          for (size_t hi = 0; hi < h; ++hi)
            for (size_t wi = 0; wi < w; ++wi) {
              const size_t src_i = ((ni * c + ci) * h + hi) * w + wi;
              const size_t dst_i = ((ni * h + hi) * w + wi) * c + ci;
              std::memcpy(dst + dst_i * elem, src + src_i * elem, elem);
            }
      input_data = result.transformed.data();
      native_shape = {shape[0], shape[2], shape[3], shape[1]};
    }
    if (native_shape.size() > RKNN_MAX_DIMS)
      throw Exception("Input rank is too large");
    result.view.data = input_data;
    result.view.bytes = value_impl->logical_bytes();
    result.view.type = type;
    result.view.n_dims = static_cast<uint32_t>(native_shape.size());
    for (size_t i = 0; i < native_shape.size(); ++i)
      result.view.dims[i] = static_cast<uint32_t>(native_shape[i]);
    return result;
  }

  std::vector<PreparedInput>
  prepare_inputs(const char *const *names, const Value *values, size_t count,
                 bool allow_batch) const {
    if (count != input_attrs.size())
      throw Exception("Input count mismatch: expected " +
                      std::to_string(input_attrs.size()) + ", got " +
                      std::to_string(count));
    if (names == nullptr || values == nullptr)
      throw Exception("input names/values must not be null");
    std::vector<const Value *> ordered(input_attrs.size(), nullptr);
    std::set<size_t> seen;
    for (size_t i = 0; i < count; ++i) {
      const size_t idx = input_index(names[i]);
      if (!seen.insert(idx).second)
        throw Exception("Duplicate input name: " + std::string(names[i]));
      ordered[idx] = &values[i];
    }
    std::vector<PreparedInput> result;
    result.reserve(ordered.size());
    for (size_t i = 0; i < ordered.size(); ++i)
      result.push_back(prepare_one(*ordered[i], i, allow_batch));
    return result;
  }

  std::future<ResultPack>
  submit(std::vector<AsyncEzRknn::InputView> views) {
    auto promise = std::make_shared<std::promise<ResultPack>>();
    auto future = promise->get_future();
    const auto deadline = std::chrono::steady_clock::now() + submit_timeout;
    begin_async_callback();
    try {
      while (true) {
        auto task = async->asyncInferenceDyn(
            [this, promise](size_t id,
                            AsyncEzRknn::InferenceResponse response) {
              struct CompletionGuard {
                Impl *impl;
                ~CompletionGuard() { impl->end_async_callback(); }
              } guard{this};
              try {
                std::vector<rknn_tensor_attr> attrs;
                if (response.error_message.empty()) {
                  attrs = async->takeTaskOutputAttrs(id);
                  if (attrs.empty())
                    attrs = output_attrs;
                }
                promise->set_value(
                    ResultPack{std::move(response), std::move(attrs)});
              } catch (...) {
                try {
                  promise->set_exception(std::current_exception());
                } catch (...) {
                }
              }
            },
            views);
        if (task)
          return future;
        if (!async->wait_for_submit_capacity_until(deadline))
          throw Exception("run submit timed out: async queue/callback pipeline "
                          "is saturated");
      }
    } catch (...) {
      end_async_callback();
      throw;
    }
  }

  std::vector<Value> values_from_result(ResultPack result,
                                        const std::vector<size_t> &indices) {
    if (!result.first.error_message.empty())
      throw Exception(result.first.error_message);
    std::vector<Value> values;
    values.reserve(indices.size());
    for (size_t idx : indices) {
      if (idx >= result.first.outputs.size() || idx >= result.second.size())
        throw Exception("Output index out of range");
      auto &buffer = result.first.outputs[idx];
      std::shared_ptr<void> owner(buffer, buffer.get());
      auto impl = RknnOrtValue::from_cpu_buffer(
          buffer.get(), result.second[idx].n_elems * sizeof(float),
          attr_shape_i64(result.second[idx]), RKNN_TENSOR_FLOAT32,
          std::move(owner));
      values.emplace_back(ValueAccess::make(std::move(impl)));
    }
    return values;
  }

  std::vector<Value> run_batch(std::vector<PreparedInput> prepared,
                               const std::vector<size_t> &indices) {
    size_t batch = 0;
    std::vector<size_t> sample_bytes(prepared.size());
    for (size_t i = 0; i < prepared.size(); ++i) {
      const auto &shape = prepared[i].owner->shape();
      const size_t current = static_cast<size_t>(shape[0]);
      if (i == 0)
        batch = current;
      else if (batch != current)
        throw Exception("Batch size mismatch across inputs");
      if (input_attrs[i].n_dims == 0 ||
          (input_attrs[i].dims[0] != 1 && input_attrs[i].dims[0] != 0))
        throw Exception("Batch mode requires model batch dimension == 1");
      sample_bytes[i] = prepared[i].view.bytes / batch;
    }
    std::vector<std::future<ResultPack>> futures;
    futures.reserve(batch);
    for (size_t b = 0; b < batch; ++b) {
      std::vector<AsyncEzRknn::InputView> views;
      views.reserve(prepared.size());
      for (size_t i = 0; i < prepared.size(); ++i) {
        auto view = prepared[i].view;
        view.data = static_cast<const uint8_t *>(prepared[i].view.data) +
                    b * sample_bytes[i];
        view.bytes = sample_bytes[i];
        view.dims[0] = 1;
        views.push_back(view);
      }
      futures.push_back(submit(std::move(views)));
    }
    std::vector<ResultPack> results;
    results.reserve(batch);
    for (auto &future : futures) {
      auto result = future.get();
      if (!result.first.error_message.empty())
        throw Exception(result.first.error_message);
      results.push_back(std::move(result));
    }
    std::vector<Value> outputs;
    outputs.reserve(indices.size());
    for (size_t idx : indices) {
      const auto &attr = results.front().second.at(idx);
      std::vector<int64_t> shape = attr_shape_i64(attr);
      if (shape.empty())
        shape = {static_cast<int64_t>(batch)};
      else
        shape[0] = static_cast<int64_t>(batch);
      const size_t sample_elements = attr.n_elems;
      const size_t total_bytes = batch * sample_elements * sizeof(float);
      auto storage = std::shared_ptr<uint8_t[]>(new uint8_t[total_bytes],
                                                 std::default_delete<uint8_t[]>());
      for (size_t b = 0; b < batch; ++b)
        std::memcpy(storage.get() + b * sample_elements * sizeof(float),
                    results[b].first.outputs.at(idx).get(),
                    sample_elements * sizeof(float));
      std::shared_ptr<void> owner(storage, storage.get());
      outputs.emplace_back(ValueAccess::make(RknnOrtValue::from_cpu_buffer(
          storage.get(), total_bytes, std::move(shape), RKNN_TENSOR_FLOAT32,
          std::move(owner))));
    }
    return outputs;
  }
};

Session::Session(const Env &env, const char *model_path,
                 const SessionOptions &options,
                 const Session *use_weight_from)
    : impl_(std::make_shared<Impl>()) {
  impl_->env = env.impl_;
  if (model_path == nullptr)
    throw Exception("model_path is null");
  const std::string path(model_path);
  const auto provider = select_model_provider(
      options.GetExecutionProvider(),
      detect_model_provider(std::filesystem::path(path)));
  if (provider == ExecutionProvider::Rknpu3) {
    if (use_weight_from)
      throw Exception("use_weight_from is supported by RKNPU2 only");
    impl_->initialize_rknpu3(options, &path);
    return;
  }
  if (use_weight_from)
    impl_->weight_master = use_weight_from->impl_;
  impl_->initialize_rknpu2(
      [impl = impl_, path](AsyncEzRknn::Layout layout,
                           std::optional<rknn_core_mask> mask,
                           bool disable_dup,
                           std::optional<rknn_context> master,
                           AsyncEzRknn::WarningHandler warning) {
        return std::make_shared<AsyncEzRknn>(
            std::filesystem::path(path), layout, impl->options.max_queue_size,
            impl->options.threads_per_core,
            impl->options.sequential_callbacks, impl->options.schedule, mask,
            impl->options.enable_pacing, disable_dup, master,
            std::move(warning));
      },
      options, impl_->weight_master);
}

Session::Session(const Env &env, const void *model_data, size_t model_size,
                 const SessionOptions &options,
                 const Session *use_weight_from)
    : impl_(std::make_shared<Impl>()) {
  impl_->env = env.impl_;
  if (model_data == nullptr || model_size == 0)
    throw Exception("model data must not be empty");
  const auto provider = select_model_provider(
      options.GetExecutionProvider(), detect_model_provider(model_data,
                                                             model_size));
  std::vector<uint8_t> bytes(model_size);
  std::memcpy(bytes.data(), model_data, model_size);
  if (provider == ExecutionProvider::Rknpu3) {
    if (use_weight_from)
      throw Exception("use_weight_from is supported by RKNPU2 only");
    impl_->initialize_rknpu3(options, nullptr, std::move(bytes));
    return;
  }
  if (use_weight_from)
    impl_->weight_master = use_weight_from->impl_;
  impl_->initialize_rknpu2(
      [impl = impl_, bytes = std::move(bytes)](
          AsyncEzRknn::Layout layout, std::optional<rknn_core_mask> mask,
          bool disable_dup, std::optional<rknn_context> master,
          AsyncEzRknn::WarningHandler warning) mutable {
        return std::make_shared<AsyncEzRknn>(
            std::move(bytes), layout, impl->options.max_queue_size,
            impl->options.threads_per_core,
            impl->options.sequential_callbacks, impl->options.schedule, mask,
            impl->options.enable_pacing, disable_dup, master,
            std::move(warning));
      },
      options, impl_->weight_master);
}

Session::~Session() {
  if (impl_)
    impl_->wait_for_async_callbacks();
}

Session &Session::operator=(Session &&other) noexcept {
  if (this == &other)
    return *this;
  if (impl_)
    impl_->wait_for_async_callbacks();
  impl_ = std::move(other.impl_);
  return *this;
}

std::vector<Value>
Session::Run(const RunOptions &run_options, const char *const *input_names,
             const Value *input_values, size_t input_count,
             const char *const *output_names, size_t output_count) {
  impl_->require_rknpu2("regular Run");
  const bool batch = run_options.GetDispatchBatch();
  auto indices = impl_->output_indices(output_names, output_count);
  auto prepared =
      impl_->prepare_inputs(input_names, input_values, input_count, batch);
  if (batch)
    return impl_->run_batch(std::move(prepared), indices);
  std::vector<AsyncEzRknn::InputView> views;
  for (const auto &item : prepared)
    views.push_back(item.view);
  return impl_->values_from_result(impl_->submit(std::move(views)).get(),
                                   indices);
}

void Session::Run(const RunOptions &run_options,
                  const char *const *input_names, const Value *input_values,
                  size_t input_count, const char *const *output_names,
                  Value *output_values, size_t output_count) {
  if (output_values == nullptr && output_count != 0)
    throw Exception("output_values is null");
  auto outputs = Run(run_options, input_names, input_values, input_count,
                     output_names, output_count);
  for (size_t i = 0; i < outputs.size(); ++i)
    output_values[i] = std::move(outputs[i]);
}

void Session::RunAsync(const RunOptions &run_options,
                       const char *const *input_names,
                       const Value *input_values, size_t input_count,
                       const char *const *output_names, Value *output_values,
                       size_t output_count, RunAsyncCallbackFn callback,
                       void *user_data) {
  impl_->require_rknpu2("RunAsync");
  if (callback == nullptr)
    throw Exception("callback must not be null");
  if (run_options.GetDispatchBatch())
    throw Exception("RunAsync does not support dispatch batch");
  if (output_values == nullptr && output_count != 0)
    throw Exception("output_values is null");
  auto indices = impl_->output_indices(output_names, output_count);
  auto prepared =
      impl_->prepare_inputs(input_names, input_values, input_count, false);
  std::vector<AsyncEzRknn::InputView> views;
  for (const auto &item : prepared)
    views.push_back(item.view);
  auto *impl = impl_.get();
  impl->begin_async_callback();
  const auto deadline = std::chrono::steady_clock::now() + impl_->submit_timeout;
  try {
    while (true) {
      auto submitted = impl_->async->asyncInferenceDyn(
          [impl, indices, output_values, output_count,
           callback, user_data](
              size_t id, AsyncEzRknn::InferenceResponse response) mutable {
            struct CompletionGuard {
              Impl *impl;
              ~CompletionGuard() { impl->end_async_callback(); }
            } guard{impl};
            try {
              std::vector<rknn_tensor_attr> attrs;
              if (response.error_message.empty()) {
                attrs = impl->async->takeTaskOutputAttrs(id);
                if (attrs.empty())
                  attrs = impl->output_attrs;
              }
              auto outputs = impl->values_from_result(
                  {std::move(response), std::move(attrs)}, indices);
              for (size_t i = 0; i < outputs.size(); ++i)
                output_values[i] = std::move(outputs[i]);
              callback(user_data, output_values, output_count, "");
            } catch (const std::exception &error) {
              callback(user_data, nullptr, 0, error.what());
            }
          },
          views);
      if (submitted)
        return;
      if (!impl_->async->wait_for_submit_capacity_until(deadline))
        throw Exception("RunAsync submit timed out");
    }
  } catch (...) {
    impl->end_async_callback();
    throw;
  }
}

std::optional<std::vector<Value>>
Session::RunPipeline(const char *const *input_names, const Value *input_values,
                     size_t input_count, size_t depth, bool reset) {
  impl_->require_rknpu2("RunPipeline");
  if (depth == 0)
    throw Exception("pipeline depth must be > 0");
  auto prepared =
      impl_->prepare_inputs(input_names, input_values, input_count, false);
  std::vector<AsyncEzRknn::InputView> views;
  for (const auto &item : prepared)
    views.push_back(item.view);
  auto future = impl_->submit(std::move(views));
  std::unique_lock<std::mutex> lock(impl_->pipeline_mutex);
  if (!impl_->pipeline_initialized || reset || depth != impl_->pipeline_depth) {
    impl_->pipeline_initialized = true;
    impl_->pipeline_depth = depth;
    while (!impl_->pipeline_queue.empty())
      impl_->pipeline_queue.pop();
  }
  impl_->pipeline_queue.push(std::move(future));
  if (impl_->pipeline_queue.size() <= depth)
    return std::nullopt;
  auto ready = std::move(impl_->pipeline_queue.front());
  impl_->pipeline_queue.pop();
  lock.unlock();
  return impl_->values_from_result(ready.get(),
                                   impl_->output_indices(nullptr, 0));
}

size_t Session::GetInputCount() const { return impl_->input_names.size(); }
size_t Session::GetOutputCount() const { return impl_->output_names.size(); }

const std::string &Session::GetInputName(size_t index) const {
  return impl_->input_names.at(index);
}

const std::string &Session::GetOutputName(size_t index) const {
  return impl_->output_names.at(index);
}

TypeInfo Session::GetInputTypeInfo(size_t index) const {
  if (impl_->provider == ExecutionProvider::Rknpu3) {
    const auto type = ztu::rk::rknpu3_attr_rknpu2_compatible_type(
        impl_->rknpu3->input_attrs().at(index));
    return TypeInfo(TensorTypeAndShapeInfo(
        from_rknn_type(type), ztu::rk::rknpu3_attr_shape(
                                  impl_->rknpu3->input_attrs().at(index))));
  }
  const auto &attr = impl_->input_attrs.at(index);
  return TypeInfo(TensorTypeAndShapeInfo(from_rknn_type(attr.type),
                                         attr_shape_i64(attr)));
}

TypeInfo Session::GetOutputTypeInfo(size_t index) const {
  if (impl_->provider == ExecutionProvider::Rknpu3) {
    const auto type = ztu::rk::rknpu3_attr_rknpu2_compatible_type(
        impl_->rknpu3->output_attrs().at(index));
    return TypeInfo(TensorTypeAndShapeInfo(
        from_rknn_type(type), ztu::rk::rknpu3_attr_shape(
                                  impl_->rknpu3->output_attrs().at(index))));
  }
  const auto &attr = impl_->output_attrs.at(index);
  return TypeInfo(TensorTypeAndShapeInfo(from_rknn_type(attr.type),
                                         attr_shape_i64(attr)));
}

std::vector<NodeArg> Session::GetInputs() const {
  if (impl_->provider == ExecutionProvider::Rknpu3)
    return build_rknpu3_node_args(impl_->rknpu3->input_attrs(),
                                  impl_->input_names);
  return build_node_args(impl_->input_attrs, impl_->input_names);
}

std::vector<NodeArg> Session::GetOutputs() const {
  if (impl_->provider == ExecutionProvider::Rknpu3)
    return build_rknpu3_node_args(impl_->rknpu3->output_attrs(),
                                  impl_->output_names);
  return build_node_args(impl_->output_attrs, impl_->output_names);
}

std::vector<NodeArg> Session::GetNativeInputs() const {
  if (impl_->provider == ExecutionProvider::Rknpu3)
    return GetInputs();
  return build_node_args(impl_->zero_copy->native_input_attrs(),
                         impl_->zero_copy->input_names());
}

std::vector<NodeArg> Session::GetNativeOutputs() const {
  if (impl_->provider == ExecutionProvider::Rknpu3)
    return GetOutputs();
  return build_node_args(impl_->zero_copy->native_output_attrs(),
                         impl_->zero_copy->output_names());
}

ModelMetadata Session::GetModelMetadata() const {
  ModelMetadata result;
  if (impl_->provider == ExecutionProvider::Rknpu3) {
    if (!impl_->rknpu3->custom_string().empty())
      result.custom_metadata_map["rknn_custom_string"] =
          impl_->rknpu3->custom_string();
    if (!impl_->rknpu3->runtime_version().empty())
      result.custom_metadata_map["rknpu3_runtime_version"] =
          impl_->rknpu3->runtime_version();
    return result;
  }
  if (!impl_->async->custom_string().empty())
    result.custom_metadata_map["rknn_custom_string"] =
        impl_->async->custom_string();
  return result;
}

const char *Session::GetExecutionProviderName() const noexcept {
  return impl_ && impl_->provider == ExecutionProvider::Rknpu3
             ? "RKNPU3ExecutionProvider"
             : "RKNPU2ExecutionProvider";
}

DeviceType Session::GetDeviceType() const noexcept {
  return impl_ && impl_->provider == ExecutionProvider::Rknpu3
             ? DeviceType::Rknpu3
             : DeviceType::Rknpu2;
}

Value Session::CreateIoValue(std::string_view name, IoKind kind,
                             const MemoryInfo &memory_info,
                             TensorElementType type,
                             std::optional<std::vector<int64_t>> shape,
                             uint64_t memory_flags) {
  if (memory_info.GetDeviceId() != 0)
    throw Exception("RKNN supports device_id 0 only");
  const std::string io_kind = kind == IoKind::Input ? "input" : "output";
  const auto native_type = to_rknn_type(type);
  if (impl_->provider == ExecutionProvider::Rknpu3) {
    if (memory_info.GetDeviceType() == DeviceType::Rknpu2)
      throw Exception("An RKNPU3 session cannot allocate an RKNPU2 OrtValue");
    if (memory_info.GetDeviceType() == DeviceType::Cpu) {
      const size_t index = kind == IoKind::Input
                               ? impl_->rknpu3->input_index(std::string(name))
                               : impl_->rknpu3->output_index(std::string(name));
      const auto &attr = kind == IoKind::Input
                             ? impl_->rknpu3->input_attrs().at(index)
                             : impl_->rknpu3->output_attrs().at(index);
      auto dims = shape.value_or(ztu::rk::rknpu3_attr_shape(attr));
      const auto actual_type =
          native_type == RKNN_TENSOR_TYPE_MAX
              ? ztu::rk::rknpu3_attr_rknpu2_compatible_type(attr)
              : native_type;
      return Value(RknnOrtValue::allocate_cpu(std::move(dims), actual_type));
    }
    return Value(impl_->rknpu3->create_value_for_io(
        std::string(name), kind == IoKind::Input, native_type, memory_flags,
        shape));
  }
  if (memory_info.GetDeviceType() == DeviceType::Rknpu3)
    throw Exception("An RKNPU2 session cannot allocate an RKNPU3 OrtValue");
  if (memory_info.GetDeviceType() == DeviceType::Cpu) {
    const auto &attr = kind == IoKind::Input
                           ? impl_->zero_copy->logical_input_attr(
                                 std::string(name))
                           : impl_->zero_copy->logical_output_attr(
                                 std::string(name));
    auto dims = shape.value_or(attr_shape_i64(attr));
    auto actual_type = native_type == RKNN_TENSOR_TYPE_MAX ? attr.type
                                                           : native_type;
    return Value(RknnOrtValue::allocate_cpu(std::move(dims), actual_type));
  }
  return Value(impl_->zero_copy->create_value_for_io(
      std::string(name), io_kind, native_type, memory_flags, shape));
}

Value Session::CreateIoValueFromDmaBuf(
    std::string_view name, IoKind kind, int fd, size_t size,
    TensorElementType type, std::optional<std::vector<int64_t>> shape,
    int32_t offset, void *virt_addr) {
  if (size > std::numeric_limits<uint32_t>::max() &&
      impl_->provider == ExecutionProvider::Rknpu2)
    throw Exception("dma-buf size is too large for RKNPU2");
  if (impl_->provider == ExecutionProvider::Rknpu3) {
    return Value(impl_->rknpu3->create_value_from_fd(
        std::string(name), kind == IoKind::Input, to_rknn_type(type), fd,
        virt_addr, size, offset, shape));
  }
  return Value(impl_->zero_copy->create_value_from_fd(
      std::string(name), kind == IoKind::Input ? "input" : "output",
      to_rknn_type(type), fd, virt_addr, static_cast<uint32_t>(size), offset,
      shape));
}

struct IoBinding::Impl {
  std::shared_ptr<Session::Impl> session;
  std::shared_ptr<RknnIoBinding> binding;
  std::shared_ptr<Rknpu3IoBinding> rknpu3_binding;
  std::vector<DeviceType> output_devices;
  std::set<size_t> requested_output_indices;
};

IoBinding::IoBinding(Session &session) : impl_(std::make_shared<Impl>()) {
  impl_->session = session.impl_;
  if (session.impl_->provider == ExecutionProvider::Rknpu3)
    impl_->rknpu3_binding =
        std::make_shared<Rknpu3IoBinding>(session.impl_->rknpu3);
  else
    impl_->binding = std::make_shared<RknnIoBinding>(session.impl_->zero_copy);
}

IoBinding::~IoBinding() = default;

void IoBinding::BindInput(const char *name, const Value &value) {
  if (name == nullptr || !value.impl_)
    throw Exception("invalid input binding");
  if (impl_->session->provider == ExecutionProvider::Rknpu3) {
    if (value.impl_->is_cpu())
      impl_->rknpu3_binding->bind_cpu_input(name, value.impl_);
    else
      impl_->rknpu3_binding->bind_ortvalue_input(name, value.impl_);
  } else {
    if (value.impl_->is_cpu())
      impl_->binding->bind_cpu_input(name, value.impl_);
    else
      impl_->binding->bind_ortvalue_input(name, value.impl_);
  }
}

void IoBinding::BindOutput(const char *name, const Value &value) {
  if (name == nullptr || !value.impl_)
    throw Exception("invalid output binding");
  if (value.impl_->is_cpu())
    throw Exception("preallocated CPU output binding is not supported; bind "
                    "MemoryInfo::CreateCpu() instead");
  const bool rknpu3 =
      impl_->session->provider == ExecutionProvider::Rknpu3;
  const size_t idx = rknpu3 ? impl_->session->rknpu3->output_index(name)
                            : impl_->session->zero_copy->output_index(name);
  if (rknpu3)
    impl_->rknpu3_binding->bind_ortvalue_output(name, value.impl_);
  else
    impl_->binding->bind_ortvalue_output(name, value.impl_);
  impl_->requested_output_indices.insert(idx);
  impl_->output_devices.push_back(rknpu3 ? DeviceType::Rknpu3
                                         : DeviceType::Rknpu2);
}

void IoBinding::BindOutput(const char *name,
                           const MemoryInfo &memory_info) {
  if (name == nullptr)
    throw Exception("output name is null");
  const bool rknpu3 =
      impl_->session->provider == ExecutionProvider::Rknpu3;
  const DeviceType expected =
      rknpu3 ? DeviceType::Rknpu3 : DeviceType::Rknpu2;
  if (memory_info.GetDeviceType() != DeviceType::Cpu &&
      memory_info.GetDeviceType() != expected)
    throw Exception("Output MemoryInfo backend does not match the session "
                    "execution provider");
  const size_t idx = rknpu3 ? impl_->session->rknpu3->output_index(name)
                            : impl_->session->zero_copy->output_index(name);
  if (rknpu3)
    impl_->rknpu3_binding->bind_output(
        name, memory_info.GetName(), memory_info.GetDeviceId(), std::nullopt,
        std::nullopt, std::nullopt, RKNN_FLAG_MEMORY_FLAGS_DEFAULT);
  else
    impl_->binding->bind_output(
        name, memory_info.GetName(), memory_info.GetDeviceId(), std::nullopt,
        std::nullopt, std::nullopt, RKNN_FLAG_MEMORY_FLAGS_DEFAULT);
  impl_->requested_output_indices.insert(idx);
  impl_->output_devices.push_back(memory_info.GetDeviceType());
}

void IoBinding::ClearBoundInputs() {
  if (impl_->rknpu3_binding)
    impl_->rknpu3_binding->clear_binding_inputs();
  else
    impl_->binding->clear_binding_inputs();
}

void IoBinding::ClearBoundOutputs() {
  if (impl_->rknpu3_binding)
    impl_->rknpu3_binding->clear_binding_outputs();
  else
    impl_->binding->clear_binding_outputs();
  impl_->output_devices.clear();
  impl_->requested_output_indices.clear();
}

void IoBinding::SynchronizeInputs() {
  if (impl_->rknpu3_binding)
    impl_->rknpu3_binding->synchronize_inputs();
  else
    impl_->binding->synchronize_inputs();
}

void IoBinding::SynchronizeOutputs() {
  if (impl_->rknpu3_binding)
    impl_->rknpu3_binding->synchronize_outputs();
  else
    impl_->binding->synchronize_outputs();
}

std::vector<Value> IoBinding::GetOutputValues() const {
  auto native = impl_->rknpu3_binding
                    ? impl_->rknpu3_binding->get_outputs()
                    : impl_->binding->get_outputs();
  std::vector<Value> result;
  const size_t requested = impl_->output_devices.size();
  result.reserve(requested);
  for (size_t i = 0; i < requested && i < native.size(); ++i) {
    if (impl_->output_devices[i] == DeviceType::Cpu) {
      auto cpu = RknnOrtValue::allocate_cpu(native[i]->shape(),
                                            native[i]->element_type());
      native[i]->copy_to(reinterpret_cast<void *>(cpu->data_ptr()),
                         cpu->logical_bytes());
      result.emplace_back(Value(std::move(cpu)));
    } else {
      result.emplace_back(Value(native[i]));
    }
  }
  return result;
}

void Session::Run(const RunOptions &run_options, IoBinding &binding) {
  (void)run_options;
  if (!binding.impl_ || binding.impl_->session.get() != impl_.get())
    throw Exception("IoBinding belongs to a different session");
  const bool rknpu3 = impl_->provider == ExecutionProvider::Rknpu3;
  for (size_t i = 0; i < impl_->output_names.size(); ++i) {
    if (!binding.impl_->requested_output_indices.count(i)) {
      if (rknpu3)
        binding.impl_->rknpu3_binding->bind_output(
            impl_->output_names[i], "rknpu3", 0, std::nullopt, std::nullopt,
            std::nullopt, RKNN_FLAG_MEMORY_FLAGS_DEFAULT);
      else
        binding.impl_->binding->bind_output(
            impl_->output_names[i], "rknpu2", 0, std::nullopt, std::nullopt,
            std::nullopt, RKNN_FLAG_MEMORY_FLAGS_DEFAULT);
    }
  }
  if (rknpu3)
    impl_->rknpu3->run_with_iobinding(*binding.impl_->rknpu3_binding);
  else
    impl_->zero_copy->run_with_iobinding(*binding.impl_->binding);
}

} // namespace ztu::somemodelruntime::rknn
