#include "rknpu3_backend.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace ztu::rk {
namespace {

using namespace rknpu3_abi;
using somemodelruntime::rknn::Rknpu3ProviderOptions;

std::string status_text(int status) {
  switch (status) {
  case kSuccess:
    return "success";
  case -1:
    return "generic failure";
  case -2:
    return "invalid argument";
  case -3:
    return "invalid model";
  case -4:
    return "invalid context";
  case -5:
    return "run task failed";
  case -6:
    return "out of memory";
  case -7:
    return "timeout";
  case -8:
    return "invalid input";
  case -9:
    return "invalid output";
  case -10:
    return "device unavailable";
  case -11:
    return "device mismatch";
  case -12:
    return "target platform mismatch";
  case -13:
    return "communication error";
  case -14:
    return "memory synchronization failed";
  case -15:
    return "invalid decryption key";
  case -16:
    return "model decryption failed";
  case -17:
    return "streaming weight load is not prepared";
  case -18:
    return "incompatible runtime/proxy/server versions";
  case kWarningCoreUnused:
    return "requested NPU core is unused";
  default:
    return "unknown status";
  }
}

std::runtime_error api_error(const char *operation, int status) {
  return std::runtime_error(std::string(operation) + " failed: " +
                            std::to_string(status) + " (" +
                            status_text(status) + ")");
}

class Rknpu3Api {
public:
  using InitFn = int (*)(Context *, InitExtend *);
  using LoadPathFn = int (*)(Context, const char *, const char *);
  using LoadDataFn = int (*)(Context, const void *, uint64_t, const void *,
                             uint64_t);
  using ModelInitFn = int (*)(Context, Config *);
  using DestroyFn = int (*)(Context);
  using QueryFn = int (*)(Context, QueryCommand, void *, uint64_t);
  using RunFn = int (*)(Context, const Tensor *, uint32_t, Tensor *, uint32_t);
  using CreateMemFn = TensorMem *(*)(Context, uint64_t, int32_t,
                                     MemoryAllocFlags);
  using CreateMemFromFdFn = TensorMem *(*)(Context, int32_t, void *, uint64_t,
                                           uint64_t);
  using DestroyMemFn = int (*)(Context, TensorMem *);
  using MemSyncFn = int (*)(Context, TensorMem *, MemorySyncMode);

  static std::shared_ptr<Rknpu3Api>
  load(const std::optional<std::string> &override_path) {
    std::vector<std::string> candidates;
    if (override_path && !override_path->empty()) {
      candidates.push_back(*override_path);
    } else if (const char *environment =
                   std::getenv("ZTU_RKNPU3_RUNTIME_LIBRARY")) {
      if (*environment != '\0')
        candidates.emplace_back(environment);
    }
    if (candidates.empty()) {
      candidates.emplace_back("librknn3_api.so");
      candidates.emplace_back("librknn3_api_rkcp.so");
      candidates.emplace_back("librknn3_api_native.so");
    }

    std::string errors;
    for (const auto &candidate : candidates) {
      dlerror();
      void *handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
      if (handle != nullptr) {
        auto result = std::shared_ptr<Rknpu3Api>(new Rknpu3Api(handle, candidate));
        result->resolve_all();
        return result;
      }
      const char *error = dlerror();
      if (error != nullptr) {
        if (!errors.empty())
          errors += "; ";
        errors += candidate + ": " + error;
      }
    }
    throw std::runtime_error(
        "unable to load the RKNPU3 runtime. Set provider option "
        "'runtime_library' or ZTU_RKNPU3_RUNTIME_LIBRARY. Tried: " +
        errors);
  }

  ~Rknpu3Api() {
    if (handle_ != nullptr)
      dlclose(handle_);
  }

  const std::string &path() const { return path_; }

  InitFn init = nullptr;
  LoadPathFn load_model_from_path = nullptr;
  LoadDataFn load_model_from_data = nullptr;
  ModelInitFn model_init = nullptr;
  DestroyFn destroy = nullptr;
  QueryFn query = nullptr;
  RunFn run = nullptr;
  CreateMemFn create_mem = nullptr;
  CreateMemFromFdFn create_mem_from_fd = nullptr;
  DestroyMemFn destroy_mem = nullptr;
  MemSyncFn mem_sync = nullptr;

private:
  Rknpu3Api(void *handle, std::string path)
      : handle_(handle), path_(std::move(path)) {}

  template <typename Fn> Fn resolve(const char *name) {
    dlerror();
    void *symbol = dlsym(handle_, name);
    const char *error = dlerror();
    if (symbol == nullptr || error != nullptr) {
      throw std::runtime_error("RKNPU3 runtime '" + path_ +
                               "' is missing symbol " + name +
                               (error ? std::string(": ") + error : ""));
    }
    return reinterpret_cast<Fn>(symbol);
  }

  void resolve_all() {
    init = resolve<InitFn>("rknn3_init");
    load_model_from_path =
        resolve<LoadPathFn>("rknn3_load_model_from_path");
    load_model_from_data =
        resolve<LoadDataFn>("rknn3_load_model_from_data");
    model_init = resolve<ModelInitFn>("rknn3_model_init");
    destroy = resolve<DestroyFn>("rknn3_destroy");
    query = resolve<QueryFn>("rknn3_query");
    run = resolve<RunFn>("rknn3_run");
    create_mem = resolve<CreateMemFn>("rknn3_create_mem");
    create_mem_from_fd =
        resolve<CreateMemFromFdFn>("rknn3_create_mem_from_fd");
    destroy_mem = resolve<DestroyMemFn>("rknn3_destroy_mem");
    mem_sync = resolve<MemSyncFn>("rknn3_mem_sync");
  }

  void *handle_ = nullptr;
  std::string path_;
};

struct ContextState {
  std::shared_ptr<Rknpu3Api> api;
  Context context = 0;
  std::mutex run_mutex;

  ~ContextState() {
    if (api && context != 0) {
      api->destroy(context);
      context = 0;
    }
  }
};

struct ValueMemoryState {
  std::shared_ptr<ContextState> context;
  TensorMem *mem = nullptr;
  TensorAttr attr = {};

  ~ValueMemoryState() {
    if (context && context->api && context->context != 0 && mem != nullptr) {
      context->api->destroy_mem(context->context, mem);
      mem = nullptr;
    }
  }

  void sync(MemorySyncMode mode) const {
    if (!context || !context->api || context->context == 0 || mem == nullptr)
      throw std::runtime_error("RKNPU3 tensor memory is no longer available");
    const int ret = context->api->mem_sync(context->context, mem, mode);
    if (ret != kSuccess)
      throw api_error("rknn3_mem_sync", ret);
  }
};

std::string fixed_string(const char *text, size_t capacity) {
  return std::string(text, strnlen(text, capacity));
}

std::vector<uint8_t> read_file(const std::filesystem::path &path,
                               const char *kind) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream)
    throw std::runtime_error(std::string("unable to open RKNPU3 ") + kind +
                             " file: " + path.string());
  const auto end = stream.tellg();
  if (end <= 0)
    throw std::runtime_error(std::string("RKNPU3 ") + kind +
                             " file is empty: " + path.string());
  std::vector<uint8_t> result(static_cast<size_t>(end));
  stream.seekg(0, std::ios::beg);
  stream.read(reinterpret_cast<char *>(result.data()),
              static_cast<std::streamsize>(result.size()));
  if (!stream)
    throw std::runtime_error(std::string("unable to read RKNPU3 ") + kind +
                             " file: " + path.string());
  return result;
}

std::filesystem::path
resolve_weight_path(const std::filesystem::path *model_path,
                    const Rknpu3ProviderOptions &options) {
  std::filesystem::path result;
  if (options.weight_path && !options.weight_path->empty()) {
    result = *options.weight_path;
  } else if (model_path != nullptr) {
    result = *model_path;
    result.replace_extension(".weight");
  } else {
    throw std::runtime_error(
        "RKNPU3 models require a weight file. For an in-memory model, set "
        "provider option 'weight_path'.");
  }
  if (!std::filesystem::is_regular_file(result)) {
    throw std::runtime_error(
        "RKNPU3 weight file not found: " + result.string() +
        ". The runtime otherwise enters streaming mode and run() can return "
        "success with incorrect output, so this provider refuses to continue.");
  }
  return result;
}

bool attr_is_dense_linear(const TensorAttr &attr) {
  if (attr.n_dims == 0)
    return true;
  if (attr.n_stride != attr.n_dims)
    return false;
  uint64_t expected = 1;
  for (size_t reverse = 0; reverse < attr.n_dims; ++reverse) {
    const size_t i = attr.n_dims - 1 - reverse;
    if (attr.stride[i] != expected)
      return false;
    if (attr.shape[i] != 0 &&
        expected > std::numeric_limits<uint64_t>::max() / attr.shape[i])
      return false;
    expected *= attr.shape[i];
  }
  return true;
}

void require_same_attr(const TensorAttr &actual, const TensorAttr &expected,
                       const std::string &context) {
  if (actual.dtype != expected.dtype || actual.layout != expected.layout ||
      actual.n_dims != expected.n_dims ||
      actual.n_stride != expected.n_stride ||
      actual.aligned_size < expected.aligned_size ||
      actual.core_id != expected.core_id)
    throw std::runtime_error(context +
                             " native RKNPU3 tensor attributes do not match");
  for (uint32_t i = 0; i < expected.n_dims; ++i)
    if (actual.shape[i] != expected.shape[i])
      throw std::runtime_error(context + " native RKNPU3 shape does not match");
  for (uint32_t i = 0; i < expected.n_stride; ++i)
    if (actual.stride[i] != expected.stride[i])
      throw std::runtime_error(context +
                               " native RKNPU3 strides do not match");
}

std::shared_ptr<ValueMemoryState>
value_memory(const std::shared_ptr<RknnOrtValue> &value,
             const std::string &context) {
  if (!value || !value->is_rknpu3())
    throw std::runtime_error(context + " requires an RKNPU3 OrtValue");
  const auto &owner = value->backend_owner();
  if (!owner)
    throw std::runtime_error(context + " has no RKNPU3 memory owner");
  return std::static_pointer_cast<ValueMemoryState>(owner);
}

} // namespace

std::vector<int64_t> rknpu3_attr_shape(const TensorAttr &attr) {
  std::vector<int64_t> result;
  result.reserve(attr.n_dims);
  for (uint32_t i = 0; i < attr.n_dims; ++i)
    result.push_back(static_cast<int64_t>(attr.shape[i]));
  return result;
}

std::vector<uint64_t> rknpu3_attr_strides(const TensorAttr &attr) {
  return std::vector<uint64_t>(attr.stride, attr.stride + attr.n_stride);
}

rknn_tensor_type rknpu3_attr_rknpu2_compatible_type(const TensorAttr &attr) {
  switch (attr.dtype) {
  case TensorType::Float32:
    return RKNN_TENSOR_FLOAT32;
  case TensorType::Float16:
    return RKNN_TENSOR_FLOAT16;
  case TensorType::Int8:
    return RKNN_TENSOR_INT8;
  case TensorType::Uint8:
    return RKNN_TENSOR_UINT8;
  case TensorType::Int16:
    return RKNN_TENSOR_INT16;
  case TensorType::Uint16:
    return RKNN_TENSOR_UINT16;
  case TensorType::Int32:
    return RKNN_TENSOR_INT32;
  case TensorType::Uint32:
    return RKNN_TENSOR_UINT32;
  case TensorType::Int64:
    return RKNN_TENSOR_INT64;
  case TensorType::Bool:
    return RKNN_TENSOR_BOOL;
  default:
    throw std::runtime_error("unsupported RKNPU3 tensor dtype: " +
                             std::to_string(static_cast<int>(attr.dtype)));
  }
}

std::string rknpu3_layout_name(TensorLayout layout) {
  switch (layout) {
  case TensorLayout::Undefined:
    return "UNDEFINED";
  case TensorLayout::Nchw:
    return "NCHW";
  case TensorLayout::Nhwc:
    return "NHWC";
  case TensorLayout::Nc1hwc2:
    return "NC1HWC2";
  case TensorLayout::Chwn:
    return "CHWN";
  case TensorLayout::Hwio:
    return "HWIO";
  case TensorLayout::Oihw:
    return "OIHW";
  case TensorLayout::O1i1hwi2o2:
    return "O1I1HWI2O2";
  default:
    return "UNKNOWN";
  }
}

struct Rknpu3Session::Impl {
  explicit Impl(const Rknpu3ProviderOptions &provider_options,
                WarningHandler handler)
      : options(provider_options), warning_handler(std::move(handler)),
        api(Rknpu3Api::load(options.runtime_library)),
        context(std::make_shared<ContextState>()) {
    context->api = api;
  }

  void initialize_context() {
    InitExtend extend = {};
    std::string device_id;
    InitExtend *extend_ptr = nullptr;
    if (options.device_id && !options.device_id->empty()) {
      device_id = *options.device_id;
      extend.device_id = device_id.data();
      extend_ptr = &extend;
    }
    const int ret = api->init(&context->context, extend_ptr);
    if (ret != kSuccess)
      throw api_error("rknn3_init", ret);
  }

  void load_path(const std::filesystem::path &model_path) {
    const auto weight_path = resolve_weight_path(&model_path, options);
    initialize_context();
    int ret = api->load_model_from_path(context->context,
                                        model_path.c_str(),
                                        weight_path.c_str());
    if (ret != kSuccess)
      throw api_error("rknn3_load_model_from_path", ret);
    initialize_model();
  }

  void load_data(std::vector<uint8_t> data) {
    const auto weight_path = resolve_weight_path(nullptr, options);
    model_data = std::move(data);
    weight_data = read_file(weight_path, "weight");
    initialize_context();
    int ret = api->load_model_from_data(
        context->context, model_data.data(), model_data.size(),
        weight_data.data(), weight_data.size());
    if (ret != kSuccess)
      throw api_error("rknn3_load_model_from_data", ret);
    initialize_model();
  }

  void initialize_model() {
    Config config = {};
    config.priority = options.priority;
    config.run_timeout = options.run_timeout_ms;
    config.run_core_mask = options.run_core_mask;
    if (config.run_core_mask == 0) {
      // TODO(rknpu3-runtime): RKNN3_NPU_CORE_AUTO is documented as 0, but
      // runtime 1.0.4 rejects it as a non-contiguous core mask. Re-evaluate
      // this workaround once the runtime accepts its documented AUTO value.
      uint32_t core_count = 0;
      const int core_ret =
          api->query(context->context, QueryCommand::CoreNumber, &core_count,
                     sizeof(core_count));
      if (core_ret != kSuccess)
        throw api_error("rknn3_query(CORE_NUMBER)", core_ret);
      if (core_count == 0 || core_count > 32)
        throw std::runtime_error("invalid RKNPU3 model core count: " +
                                 std::to_string(core_count));
      config.run_core_mask =
          core_count == 32 ? std::numeric_limits<uint32_t>::max()
                           : (uint32_t{1} << core_count) - 1;
    }
    const int init_ret = api->model_init(context->context, &config);
    if (init_ret != kSuccess && init_ret != kWarningCoreUnused)
      throw api_error("rknn3_model_init", init_ret);
    if (init_ret == kWarningCoreUnused && warning_handler)
      warning_handler("rknn3_model_init: requested NPU core is unused");

    SdkVersion version = {};
    if (api->query(context->context, QueryCommand::SdkVersion, &version,
                   sizeof(version)) == kSuccess) {
      runtime_version = fixed_string(version.api_version,
                                     sizeof(version.api_version));
    }

    InputOutputNum io = {};
    const int io_ret = api->query(context->context, QueryCommand::InOutNum, &io,
                                  sizeof(io));
    if (io_ret != kSuccess)
      throw api_error("rknn3_query(IN_OUT_NUM)", io_ret);
    query_attrs(QueryCommand::InputAttr, io.n_input, input_attrs, input_names,
                "input");
    query_attrs(QueryCommand::OutputAttr, io.n_output, output_attrs,
                output_names, "output");

    // TODO(rknpu3-runtime): QueryCommand::CustomString is public in the 1.0.4
    // header (command 7), but the RKCP 1.0.4 runtime returns unsupported and
    // emits an error-level log. Populate custom metadata once it is usable.
  }

  void query_attrs(QueryCommand command, uint32_t count,
                   std::vector<TensorAttr> &attrs,
                   std::vector<std::string> &names, const char *kind) {
    attrs.resize(count);
    names.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
      attrs[i] = {};
      attrs[i].index = i;
      const int ret = api->query(context->context, command, &attrs[i],
                                 sizeof(TensorAttr));
      if (ret != kSuccess)
        throw api_error((std::string("rknn3_query(") + kind + " attr " +
                         std::to_string(i) + ")")
                            .c_str(),
                        ret);
      if (attrs[i].n_dims > kMaxDims ||
          attrs[i].n_stride > kMaxStrideDims)
        throw std::runtime_error(std::string("invalid RKNPU3 ") + kind +
                                 " tensor rank/stride rank");
      if (attrs[i].aligned_size == 0)
        throw std::runtime_error(std::string("RKNPU3 ") + kind +
                                 " tensor has aligned_size=0");
      std::string name = fixed_string(attrs[i].name, sizeof(attrs[i].name));
      if (name.empty())
        name = std::string(kind) + "_" + std::to_string(i);
      names.push_back(std::move(name));
    }
  }

  size_t lookup(const std::vector<std::string> &names, const std::string &name,
                const char *kind) const {
    const auto it = std::find(names.begin(), names.end(), name);
    if (it == names.end())
      throw std::runtime_error("Unknown " + std::string(kind) + " name: " +
                               name);
    return static_cast<size_t>(std::distance(names.begin(), it));
  }

  std::shared_ptr<RknnOrtValue>
  make_value(const TensorAttr &attr, rknn_tensor_type requested_type,
             uint64_t alloc_flags, const std::vector<int64_t> &shape,
             TensorMem *mem) {
    if (mem == nullptr)
      throw std::runtime_error("RKNPU3 tensor memory allocation/import failed");
    auto state = std::make_shared<ValueMemoryState>();
    state->context = context;
    state->mem = mem;
    state->attr = attr;

    const rknn_tensor_type actual_type =
        rknpu3_attr_rknpu2_compatible_type(attr);
    if (requested_type != RKNN_TENSOR_TYPE_MAX && requested_type != actual_type)
      throw std::runtime_error("RKNPU3 OrtValue element type mismatch");
    if ((alloc_flags & ~uint64_t{3}) != 0)
      throw std::runtime_error(
          "RKNPU3 supports cacheable/non-cacheable memory flags only");
    if (mem->virt_addr == nullptr || mem->size < attr.aligned_size) {
      throw std::runtime_error("RKNPU3 tensor memory is unmapped or too small");
    }
    if (mem->offset >
        static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
      throw std::runtime_error("RKNPU3 tensor memory offset is too large");
    }

    RknnMemoryInfo info;
    info.device_type = "rknpu3";
    info.device_id = 0;
    info.size = static_cast<size_t>(mem->size);
    info.fd = mem->fd;
    info.offset = static_cast<int32_t>(mem->offset);
    info.virt_addr = reinterpret_cast<uintptr_t>(mem->virt_addr);
    info.import_count = 1;
    info.dmabuf_backed = mem->fd >= 0;
    info.attr_size = attr.n_elems * rknn_tensor_type_size(actual_type);
    info.size_with_stride =
        attr.aligned_size <= std::numeric_limits<uint32_t>::max()
            ? static_cast<uint32_t>(attr.aligned_size)
            : 0;
    info.fmt = static_cast<int>(attr.layout);
    info.aligned_size = attr.aligned_size;
    info.core_id = attr.core_id;
    info.layout = rknpu3_layout_name(attr.layout);
    info.strides = rknpu3_attr_strides(attr);

    return RknnOrtValue::from_external_accelerator(
        "rknpu3", mem->virt_addr, static_cast<size_t>(mem->size), mem->fd,
        static_cast<int32_t>(mem->offset), shape, actual_type, state,
        [state] { state->sync(MemorySyncMode::ToDevice); },
        [state] { state->sync(MemorySyncMode::FromDevice); }, std::move(info),
        attr_is_dense_linear(attr));
  }

  Rknpu3ProviderOptions options;
  WarningHandler warning_handler;
  std::shared_ptr<Rknpu3Api> api;
  std::shared_ptr<ContextState> context;
  std::vector<uint8_t> model_data;
  std::vector<uint8_t> weight_data;
  std::vector<TensorAttr> input_attrs;
  std::vector<TensorAttr> output_attrs;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::string custom_string;
  std::string runtime_version;
};

Rknpu3Session::Rknpu3Session(const std::string &model_path,
                             const Rknpu3ProviderOptions &options,
                             WarningHandler warning_handler)
    : impl_(std::make_shared<Impl>(options, std::move(warning_handler))) {
  impl_->load_path(std::filesystem::path(model_path));
}

Rknpu3Session::Rknpu3Session(std::vector<uint8_t> model_data,
                             const Rknpu3ProviderOptions &options,
                             WarningHandler warning_handler)
    : impl_(std::make_shared<Impl>(options, std::move(warning_handler))) {
  if (model_data.empty())
    throw std::runtime_error("RKNPU3 model data must not be empty");
  impl_->load_data(std::move(model_data));
}

Rknpu3Session::~Rknpu3Session() = default;

const std::vector<TensorAttr> &Rknpu3Session::input_attrs() const {
  return impl_->input_attrs;
}
const std::vector<TensorAttr> &Rknpu3Session::output_attrs() const {
  return impl_->output_attrs;
}
const std::vector<std::string> &Rknpu3Session::input_names() const {
  return impl_->input_names;
}
const std::vector<std::string> &Rknpu3Session::output_names() const {
  return impl_->output_names;
}
const std::string &Rknpu3Session::custom_string() const {
  return impl_->custom_string;
}
const std::string &Rknpu3Session::runtime_version() const {
  return impl_->runtime_version;
}

size_t Rknpu3Session::input_index(const std::string &name) const {
  return impl_->lookup(impl_->input_names, name, "input");
}
size_t Rknpu3Session::output_index(const std::string &name) const {
  return impl_->lookup(impl_->output_names, name, "output");
}

std::shared_ptr<RknnOrtValue> Rknpu3Session::create_value_for_io(
    const std::string &name, bool is_input, rknn_tensor_type requested_type,
    uint64_t alloc_flags,
    const std::optional<std::vector<int64_t>> &requested_shape) {
  const size_t index = is_input ? input_index(name) : output_index(name);
  const TensorAttr &attr =
      is_input ? impl_->input_attrs.at(index) : impl_->output_attrs.at(index);
  const auto shape = rknpu3_attr_shape(attr);
  if (requested_shape && *requested_shape != shape)
    throw std::runtime_error("RKNPU3 OrtValue shape mismatch for " + name);
  if ((alloc_flags & ~uint64_t{3}) != 0)
    throw std::runtime_error(
        "RKNPU3 supports cacheable/non-cacheable memory flags only");
  TensorMem *mem = impl_->api->create_mem(
      impl_->context->context, attr.aligned_size, attr.core_id,
      static_cast<MemoryAllocFlags>(alloc_flags));
  return impl_->make_value(attr, requested_type, alloc_flags, shape, mem);
}

std::shared_ptr<RknnOrtValue> Rknpu3Session::create_value_from_fd(
    const std::string &name, bool is_input, rknn_tensor_type requested_type,
    int fd, void *virt_addr, uint64_t size, int32_t offset,
    const std::optional<std::vector<int64_t>> &requested_shape) {
  const size_t index = is_input ? input_index(name) : output_index(name);
  const TensorAttr &attr =
      is_input ? impl_->input_attrs.at(index) : impl_->output_attrs.at(index);
  const auto shape = rknpu3_attr_shape(attr);
  if (requested_shape && *requested_shape != shape)
    throw std::runtime_error("RKNPU3 OrtValue shape mismatch for " + name);
  if (fd < 0 || offset < 0 || size < attr.aligned_size)
    throw std::runtime_error("invalid or too-small dma-buf for RKNPU3 tensor");
  TensorMem *mem = impl_->api->create_mem_from_fd(
      impl_->context->context, fd, virt_addr, size,
      static_cast<uint64_t>(offset));
  return impl_->make_value(attr, requested_type, 0, shape, mem);
}

Rknpu3IoBinding::Rknpu3IoBinding(std::shared_ptr<Rknpu3Session> session)
    : session_(std::move(session)) {
  if (!session_)
    throw std::runtime_error("RKNPU3 IOBinding requires a valid session");
}

void Rknpu3IoBinding::bind_cpu_input(
    const std::string &name, std::shared_ptr<RknnOrtValue> cpu_value) {
  if (!cpu_value || !cpu_value->is_cpu())
    throw std::runtime_error("bind_cpu_input requires a CPU OrtValue");
  auto value = session_->create_value_for_io(
      name, true, RKNN_TENSOR_TYPE_MAX, RKNN_FLAG_MEMORY_FLAGS_DEFAULT);
  value->update_inplace(reinterpret_cast<const void *>(cpu_value->data_ptr()),
                        cpu_value->logical_bytes(), cpu_value->shape(),
                        cpu_value->element_type());
  bind_ortvalue_input(name, std::move(value));
}

void Rknpu3IoBinding::bind_ortvalue_input(
    const std::string &name, std::shared_ptr<RknnOrtValue> value) {
  const size_t index = session_->input_index(name);
  auto memory = value_memory(value, "bind_ortvalue_input");
  if (memory->context.get() != session_->impl_->context.get())
    throw std::runtime_error(
        "RKNPU3 OrtValue belongs to a different session/context");
  require_same_attr(memory->attr, session_->impl_->input_attrs.at(index),
                    "Input '" + name + "'");
  bound_inputs_[index] = std::move(value);
}

void Rknpu3IoBinding::bind_ortvalue_output(
    const std::string &name, std::shared_ptr<RknnOrtValue> value) {
  const size_t index = session_->output_index(name);
  auto memory = value_memory(value, "bind_ortvalue_output");
  if (memory->context.get() != session_->impl_->context.get())
    throw std::runtime_error(
        "RKNPU3 OrtValue belongs to a different session/context");
  require_same_attr(memory->attr, session_->impl_->output_attrs.at(index),
                    "Output '" + name + "'");
  bound_outputs_[index] = std::move(value);
  remember_output_order(index);
}

void Rknpu3IoBinding::bind_output(
    const std::string &name, const std::string &device_type, int device_id,
    std::optional<rknn_tensor_type> element_type,
    const std::optional<std::vector<int64_t>> &shape,
    std::optional<uint64_t> buffer_ptr, uint64_t alloc_flags) {
  if (device_id != 0)
    throw std::runtime_error("RKNPU3 supports device_id 0 only");
  if (buffer_ptr)
    throw std::runtime_error(
        "bind_output with raw buffer_ptr is not implemented; use an OrtValue");
  if (device_type != "cpu" && device_type != "rknpu3")
    throw std::runtime_error("device_type must be 'cpu' or 'rknpu3'");
  auto value = session_->create_value_for_io(
      name, false, element_type.value_or(RKNN_TENSOR_TYPE_MAX), alloc_flags,
      shape);
  const size_t index = session_->output_index(name);
  bound_outputs_[index] = std::move(value);
  if (device_type == "cpu")
    cpu_output_indices_[index] = true;
  remember_output_order(index);
}

std::vector<std::shared_ptr<RknnOrtValue>>
Rknpu3IoBinding::get_outputs() const {
  std::vector<std::shared_ptr<RknnOrtValue>> result;
  result.reserve(output_order_.size());
  for (size_t index : output_order_) {
    const auto it = bound_outputs_.find(index);
    if (it != bound_outputs_.end())
      result.push_back(it->second);
  }
  return result;
}

void Rknpu3IoBinding::clear_binding_inputs() { bound_inputs_.clear(); }
void Rknpu3IoBinding::clear_binding_outputs() {
  bound_outputs_.clear();
  cpu_output_indices_.clear();
  output_order_.clear();
}
void Rknpu3IoBinding::synchronize_inputs() {
  for (auto &item : bound_inputs_)
    item.second->sync_to_device();
}
void Rknpu3IoBinding::synchronize_outputs() {
  for (auto &item : bound_outputs_)
    item.second->sync_from_device();
}
void Rknpu3IoBinding::remember_output_order(size_t index) {
  if (std::find(output_order_.begin(), output_order_.end(), index) ==
      output_order_.end())
    output_order_.push_back(index);
}

void Rknpu3Session::run_with_iobinding(Rknpu3IoBinding &binding) {
  if (binding.session_.get() != this)
    throw std::runtime_error("IOBinding belongs to a different RKNPU3 session");
  if (binding.bound_inputs_.size() != impl_->input_attrs.size())
    throw std::runtime_error(
        "All RKNPU3 model inputs must be bound before run_with_iobinding");
  if (binding.bound_outputs_.size() != impl_->output_attrs.size())
    throw std::runtime_error(
        "All RKNPU3 model outputs must be bound before run_with_iobinding");

  std::vector<Tensor> inputs(impl_->input_attrs.size());
  std::vector<Tensor> outputs(impl_->output_attrs.size());
  std::vector<std::shared_ptr<ValueMemoryState>> input_memory;
  std::vector<std::shared_ptr<ValueMemoryState>> output_memory;
  input_memory.reserve(inputs.size());
  output_memory.reserve(outputs.size());

  for (size_t i = 0; i < inputs.size(); ++i) {
    auto memory = value_memory(binding.bound_inputs_.at(i), "bound input");
    memory->sync(MemorySyncMode::ToDevice);
    inputs[i] = Tensor{memory->mem, &memory->attr};
    input_memory.push_back(std::move(memory));
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto memory = value_memory(binding.bound_outputs_.at(i), "bound output");
    outputs[i] = Tensor{memory->mem, &memory->attr};
    output_memory.push_back(std::move(memory));
  }

  std::lock_guard<std::mutex> lock(impl_->context->run_mutex);
  const int ret = impl_->api->run(
      impl_->context->context, inputs.data(), static_cast<uint32_t>(inputs.size()),
      outputs.data(), static_cast<uint32_t>(outputs.size()));
  if (ret != kSuccess)
    throw api_error("rknn3_run", ret);
}

} // namespace ztu::rk
