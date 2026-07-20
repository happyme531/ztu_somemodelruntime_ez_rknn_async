#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ztu/somemodelruntime_rknn.hpp>

#include "rknn_api.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;
namespace smr = ztu::somemodelruntime::rknn;

namespace {

void emit_python_warning(std::string_view message) {
  py::gil_scoped_acquire gil;
  py::module_ warnings = py::module_::import("warnings");
  py::object user_warning = py::module_::import("builtins").attr("UserWarning");
  warnings.attr("warn")(py::str(message), user_warning, 2);
}

std::string trim(std::string value) {
  const auto first = std::find_if_not(value.begin(), value.end(), [](char ch) {
    return std::isspace(static_cast<unsigned char>(ch));
  });
  const auto last = std::find_if_not(value.rbegin(), value.rend(), [](char ch) {
                      return std::isspace(static_cast<unsigned char>(ch));
                    }).base();
  if (first >= last)
    return {};
  return std::string(first, last);
}

int64_t parse_int(py::handle value, const char *name) {
  if (py::isinstance<py::int_>(value))
    return py::cast<int64_t>(value);
  if (py::isinstance<py::str>(value)) {
    std::string text = trim(py::cast<std::string>(value));
    size_t used = 0;
    try {
      const long long result = std::stoll(text, &used);
      if (used == text.size())
        return static_cast<int64_t>(result);
    } catch (...) {
    }
  }
  throw std::runtime_error(std::string(name) + " must be an integer");
}

bool parse_bool(py::handle value, const char *name) {
  if (py::isinstance<py::bool_>(value))
    return py::cast<bool>(value);
  if (py::isinstance<py::int_>(value))
    return py::cast<int64_t>(value) != 0;
  if (py::isinstance<py::str>(value)) {
    std::string text = trim(py::cast<std::string>(value));
    std::transform(text.begin(), text.end(), text.begin(), [](char ch) {
      return static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    });
    if (text == "1" || text == "true" || text == "on" || text == "yes")
      return true;
    if (text == "0" || text == "false" || text == "off" || text == "no")
      return false;
  }
  throw std::runtime_error(std::string(name) + " must be a boolean");
}

std::vector<int64_t> parse_shape(py::handle value, bool allow_none = false) {
  if (allow_none && value.is_none())
    return {};
  if (!py::isinstance<py::sequence>(value))
    throw std::runtime_error("shape must be a sequence of integers");
  std::vector<int64_t> result;
  for (py::handle item : value.cast<py::sequence>()) {
    const int64_t dim = parse_int(item, "shape dim");
    if (dim < 0)
      throw std::runtime_error("shape dims must be >= 0");
    result.push_back(dim);
  }
  return result;
}

std::vector<int64_t> array_shape(const py::array &array) {
  std::vector<int64_t> result;
  result.reserve(static_cast<size_t>(array.ndim()));
  for (ssize_t i = 0; i < array.ndim(); ++i)
    result.push_back(static_cast<int64_t>(array.shape(i)));
  return result;
}

smr::TensorElementType dtype_to_element(const py::dtype &dtype) {
  if (dtype.is(py::dtype::of<float>()))
    return smr::TensorElementType::Float32;
  if (dtype.is(py::dtype("float16")))
    return smr::TensorElementType::Float16;
  if (dtype.is(py::dtype::of<int8_t>()))
    return smr::TensorElementType::Int8;
  if (dtype.is(py::dtype::of<uint8_t>()))
    return smr::TensorElementType::Uint8;
  if (dtype.is(py::dtype::of<int16_t>()))
    return smr::TensorElementType::Int16;
  if (dtype.is(py::dtype::of<uint16_t>()))
    return smr::TensorElementType::Uint16;
  if (dtype.is(py::dtype::of<int32_t>()))
    return smr::TensorElementType::Int32;
  if (dtype.is(py::dtype::of<uint32_t>()))
    return smr::TensorElementType::Uint32;
  if (dtype.is(py::dtype::of<int64_t>()))
    return smr::TensorElementType::Int64;
  if (dtype.is(py::dtype::of<bool>()))
    return smr::TensorElementType::Bool;
  throw std::runtime_error("Unsupported numpy dtype for RKNN tensor");
}

py::dtype element_to_dtype(smr::TensorElementType type) {
  switch (type) {
  case smr::TensorElementType::Float32:
    return py::dtype::of<float>();
  case smr::TensorElementType::Float16:
    return py::dtype("float16");
  case smr::TensorElementType::Int8:
    return py::dtype::of<int8_t>();
  case smr::TensorElementType::Uint8:
    return py::dtype::of<uint8_t>();
  case smr::TensorElementType::Int16:
    return py::dtype::of<int16_t>();
  case smr::TensorElementType::Uint16:
    return py::dtype::of<uint16_t>();
  case smr::TensorElementType::Int32:
    return py::dtype::of<int32_t>();
  case smr::TensorElementType::Uint32:
    return py::dtype::of<uint32_t>();
  case smr::TensorElementType::Int64:
    return py::dtype::of<int64_t>();
  case smr::TensorElementType::Bool:
    return py::dtype::of<bool>();
  default:
    throw std::runtime_error("Unsupported RKNN tensor type");
  }
}

std::string element_to_onnx_string(smr::TensorElementType type) {
  switch (type) {
  case smr::TensorElementType::Float32:
    return "tensor(float)";
  case smr::TensorElementType::Float16:
    return "tensor(float16)";
  case smr::TensorElementType::Int8:
    return "tensor(int8)";
  case smr::TensorElementType::Uint8:
    return "tensor(uint8)";
  case smr::TensorElementType::Int16:
    return "tensor(int16)";
  case smr::TensorElementType::Uint16:
    return "tensor(uint16)";
  case smr::TensorElementType::Int32:
    return "tensor(int32)";
  case smr::TensorElementType::Uint32:
    return "tensor(uint32)";
  case smr::TensorElementType::Int64:
    return "tensor(int64)";
  case smr::TensorElementType::Bool:
    return "tensor(bool)";
  default:
    return "tensor(unknown)";
  }
}

smr::TensorElementType dtype_like_to_element(py::handle value,
                                             bool allow_none = false) {
  if (allow_none && value.is_none())
    return smr::TensorElementType::Unknown;
  return dtype_to_element(
      py::dtype::from_args(py::reinterpret_borrow<py::object>(value)));
}

uint64_t parse_memory_flags(py::handle value) {
  if (value.is_none())
    return RKNN_FLAG_MEMORY_FLAGS_DEFAULT;
  if (py::isinstance<py::int_>(value))
    return static_cast<uint64_t>(py::cast<int64_t>(value));
  if (!py::isinstance<py::str>(value))
    throw std::runtime_error("mem_flags must be a string, integer, or None");
  std::string text = trim(py::cast<std::string>(value));
  std::transform(text.begin(), text.end(), text.begin(), [](char ch) {
    return static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  });
  if (text == "default" || text == "cacheable")
    return RKNN_FLAG_MEMORY_FLAGS_DEFAULT;
  if (text == "non_cacheable" || text == "non-cacheable" ||
      text == "uncached")
    return RKNN_FLAG_MEMORY_NON_CACHEABLE;
  if (text == "try_sram" || text == "sram")
    return RKNN_FLAG_MEMORY_TRY_ALLOC_SRAM;
  throw std::runtime_error("invalid mem_flags");
}

std::string parse_path(py::handle value, const char *name) {
  try {
    return py::cast<std::string>(py::module_::import("os").attr("fspath")(value));
  } catch (...) {
    throw std::runtime_error(std::string(name) +
                             " must be a path-like object or string");
  }
}

std::vector<std::string> parse_paths(py::handle value, const char *name) {
  if (py::isinstance<py::str>(value) || py::hasattr(value, "__fspath__"))
    return {parse_path(value, name)};
  if (!py::isinstance<py::sequence>(value))
    throw std::runtime_error(std::string(name) + " must be a path or sequence");
  std::vector<std::string> result;
  for (py::handle item : value.cast<py::sequence>())
    result.push_back(parse_path(item, name));
  return result;
}

std::vector<uint64_t> parse_schedule(py::handle value) {
  std::vector<uint64_t> result;
  if (py::isinstance<py::int_>(value)) {
    const int64_t item = parse_int(value, "schedule");
    if (item < 0)
      throw std::runtime_error("schedule values must be >= 0");
    result.push_back(static_cast<uint64_t>(item));
  } else if (py::isinstance<py::str>(value)) {
    std::string text = py::cast<std::string>(value);
    for (char &ch : text)
      if (ch == ',' || ch == '[' || ch == ']')
        ch = ' ';
    std::stringstream stream(text);
    int64_t item;
    while (stream >> item) {
      if (item < 0)
        throw std::runtime_error("schedule values must be >= 0");
      result.push_back(static_cast<uint64_t>(item));
    }
  } else if (py::isinstance<py::sequence>(value)) {
    for (py::handle entry : value.cast<py::sequence>()) {
      const int64_t item = parse_int(entry, "schedule");
      if (item < 0)
        throw std::runtime_error("schedule values must be >= 0");
      result.push_back(static_cast<uint64_t>(item));
    }
  } else {
    throw std::runtime_error("schedule must be an integer, string, or sequence");
  }
  if (result.empty())
    throw std::runtime_error("schedule must not be empty");
  return result;
}

smr::CoreMask parse_core_mask(py::handle value) {
  if (!py::isinstance<py::str>(value))
    throw std::runtime_error("tp_mode must be a string");
  std::string text = trim(py::cast<std::string>(value));
  if (text == "auto")
    return smr::CoreMask::Auto;
  if (text == "all")
    return smr::CoreMask::All;
  if (text == "0")
    return smr::CoreMask::Core0;
  if (text == "1")
    return smr::CoreMask::Core1;
  if (text == "2")
    return smr::CoreMask::Core2;
  if (text == "0,1")
    return smr::CoreMask::Core01;
  if (text == "0,1,2")
    return smr::CoreMask::Core012;
  throw std::runtime_error("invalid tp_mode");
}

py::dict extract_provider_options(py::handle object) {
  if (object.is_none())
    return py::dict();
  if (py::isinstance<py::dict>(object))
    return py::reinterpret_borrow<py::dict>(object);
  if (py::isinstance<py::sequence>(object)) {
    py::sequence sequence = py::reinterpret_borrow<py::sequence>(object);
    if (sequence.size() == 0)
      return py::dict();
    if (py::isinstance<py::dict>(sequence[0]))
      return sequence[0].cast<py::dict>();
  }
  throw std::runtime_error(
      "provider_options must be a dict or a sequence whose first item is a dict");
}

smr::RknnProviderOptions parse_provider_options(py::handle object) {
  smr::RknnProviderOptions options;
  for (auto item : extract_provider_options(object)) {
    const std::string key = py::cast<std::string>(item.first);
    py::handle value = item.second;
    if (key == "layout") {
      std::string layout = py::cast<std::string>(value);
      std::transform(layout.begin(), layout.end(), layout.begin(), ::tolower);
      if (layout == "nchw" || layout == "original")
        options.layout = smr::Layout::Nchw;
      else if (layout == "nchw_software" || layout == "original_software")
        options.layout = smr::Layout::NchwSoftware;
      else if (layout == "nhwc")
        options.layout = smr::Layout::Nhwc;
      else if (layout == "any")
        options.layout = smr::Layout::Any;
      else
        throw std::runtime_error("Unknown layout: " + layout);
    } else if (key == "zero_copy_layout") {
      std::string layout = py::cast<std::string>(value);
      if (layout == "nhwc")
        options.zero_copy_layout = smr::ZeroCopyLayout::Nhwc;
      else if (layout == "native")
        options.zero_copy_layout = smr::ZeroCopyLayout::Native;
      else
        throw std::runtime_error("Unknown zero_copy_layout: " + layout);
    } else if (key == "max_queue_size") {
      options.max_queue_size = static_cast<size_t>(parse_int(value, key.c_str()));
    } else if (key == "threads_per_core") {
      options.threads_per_core = static_cast<int>(parse_int(value, key.c_str()));
    } else if (key == "submit_timeout_ms") {
      options.submit_timeout_ms = parse_int(value, key.c_str());
    } else if (key == "sequential_callbacks") {
      options.sequential_callbacks = parse_bool(value, key.c_str());
    } else if (key == "schedule") {
      options.schedule = parse_schedule(value);
    } else if (key == "tp_mode") {
      options.tp_mode = parse_core_mask(value);
    } else if (key == "enable_pacing") {
      options.enable_pacing = parse_bool(value, key.c_str());
    } else if (key == "disable_dup_context") {
      options.disable_dup_context = parse_bool(value, key.c_str());
    } else if (key == "custom_op_path" || key == "custom_op_paths") {
      auto paths = parse_paths(value, key.c_str());
      options.custom_op_paths.insert(options.custom_op_paths.end(), paths.begin(),
                                     paths.end());
    } else if (key == "custom_op_default_path" ||
               key == "load_custom_ops_from_default_path") {
      options.custom_op_default_path = parse_bool(value, key.c_str());
    } else {
      throw std::runtime_error("Unknown provider_options key: " + key);
    }
  }
  return options;
}

smr::RunOptions parse_run_options(py::handle object) {
  smr::RunOptions result;
  constexpr const char *key = "ztu_modelrt_dispatch_batch";
  if (object.is_none())
    return result;
  if (py::isinstance<py::dict>(object)) {
    py::dict values = py::reinterpret_borrow<py::dict>(object);
    if (values.contains(py::str(key)))
      result.SetDispatchBatch(parse_bool(values[py::str(key)], key));
    return result;
  }
  if (py::hasattr(object, "get_run_config_entry")) {
    try {
      result.SetDispatchBatch(
          parse_bool(object.attr("get_run_config_entry")(key), key));
    } catch (const py::error_already_set &) {
      PyErr_Clear();
    }
  }
  return result;
}

class PyOrtValue {
public:
  explicit PyOrtValue(smr::Value value, py::object owner = py::none())
      : value_(std::move(value)), owner_(std::move(owner)) {}

  smr::Value &value() { return value_; }
  const smr::Value &value() const { return value_; }

  py::array numpy() const {
    const auto info = value_.GetTensorTypeAndShapeInfo();
    py::array result(element_to_dtype(info.GetElementType()), info.GetShape());
    value_.CopyTo(result.mutable_data(), static_cast<size_t>(result.nbytes()));
    return result;
  }

  void update(py::array source) {
    py::array contiguous = py::array::ensure(source, py::array::c_style);
    if (!contiguous)
      throw std::runtime_error("source must be a contiguous numpy array");
    value_.Update(contiguous.data(), static_cast<size_t>(contiguous.nbytes()),
                  array_shape(contiguous), dtype_to_element(contiguous.dtype()));
  }

private:
  smr::Value value_;
  py::object owner_;
};

py::array ensure_contiguous(py::handle object) {
  py::array array = py::array::ensure(object, py::array::c_style);
  if (!array)
    throw std::runtime_error("Input must be a numpy array");
  return array;
}

smr::Value value_from_array(py::array array) {
  const auto shape = array_shape(array);
  return smr::Value::CreateTensor(
      smr::MemoryInfo::CreateCpu(), array.mutable_data(),
      static_cast<size_t>(array.nbytes()), shape.data(), shape.size(),
      dtype_to_element(array.dtype()));
}

std::shared_ptr<PyOrtValue> wrap_value(smr::Value value) {
  return std::make_shared<PyOrtValue>(std::move(value));
}

struct PreparedCall {
  std::vector<std::string> names;
  std::vector<const char *> name_ptrs;
  std::vector<py::array> arrays;
  std::vector<smr::Value> values;
};

class PySession;

class PyIOBinding {
public:
  explicit PyIOBinding(smr::Session &session)
      : session_(&session), binding_(std::make_unique<smr::IoBinding>(session)) {}

  smr::IoBinding &binding() { return *binding_; }
  void bind_cpu_input(const std::string &name, py::array array) {
    auto contiguous = ensure_contiguous(array);
    auto value = value_from_array(contiguous);
    binding_->BindInput(name.c_str(), value);
  }
  void bind_value_input(const std::string &name,
                        const std::shared_ptr<PyOrtValue> &value) {
    binding_->BindInput(name.c_str(), value->value());
  }
  void bind_value_output(const std::string &name,
                         const std::shared_ptr<PyOrtValue> &value) {
    binding_->BindOutput(name.c_str(), value->value());
  }
  void bind_output(const std::string &name, const std::string &device_type,
                   int device_id, smr::TensorElementType element_type,
                   std::optional<std::vector<int64_t>> shape,
                   uint64_t memory_flags) {
    const auto info = device_type == "cpu"
                          ? smr::MemoryInfo::CreateCpu(device_id)
                          : smr::MemoryInfo::CreateRknpu2(device_id);
    if (device_type != "cpu" && device_type != "rknpu2")
      throw std::runtime_error("device_type must be 'cpu' or 'rknpu2'");
    if (element_type != smr::TensorElementType::Unknown || shape.has_value()) {
      auto validation = session_->CreateIoValue(
          name, smr::IoKind::Output, smr::MemoryInfo::CreateRknpu2(),
          element_type, shape, memory_flags);
      (void)validation;
    }
    binding_->BindOutput(name.c_str(), info);
  }
  std::vector<std::shared_ptr<PyOrtValue>> outputs() const {
    auto values = binding_->GetOutputValues();
    std::vector<std::shared_ptr<PyOrtValue>> result;
    result.reserve(values.size());
    for (auto &value : values)
      result.push_back(wrap_value(std::move(value)));
    return result;
  }
  py::list copy_outputs() const {
    py::list result;
    for (const auto &value : outputs())
      result.append(value->numpy());
    return result;
  }

private:
  smr::Session *session_;
  std::unique_ptr<smr::IoBinding> binding_;
};

class PySession {
public:
  PySession(py::handle path_or_bytes, py::handle provider_options,
            PySession *weight_master)
      : env_("ztu_somemodelruntime_ez_rknn_async") {
    env_.SetWarningHandler(emit_python_warning);
    smr::SessionOptions options;
    options.SetRknnProviderOptions(parse_provider_options(provider_options));
    if (py::isinstance<py::bytes>(path_or_bytes)) {
      std::string bytes = py::cast<std::string>(path_or_bytes);
      session_ = std::make_shared<smr::Session>(
          env_, bytes.data(), bytes.size(), options,
          weight_master ? weight_master->session_.get() : nullptr);
    } else {
      const std::string path = parse_path(path_or_bytes, "path_or_bytes");
      session_ = std::make_shared<smr::Session>(
          env_, path.c_str(), options,
          weight_master ? weight_master->session_.get() : nullptr);
    }
  }

  ~PySession() {
    if (session_) {
      py::gil_scoped_release release;
      session_.reset();
    }
  }

  smr::Session &session() { return *session_; }

  PreparedCall prepare_inputs(py::handle input_feed) {
    PreparedCall call;
    const size_t count = session_->GetInputCount();
    call.names.reserve(count);
    call.arrays.reserve(count);
    if (py::isinstance<py::array>(input_feed)) {
      if (count != 1)
        throw std::runtime_error("Model expects multiple inputs");
      call.names.push_back(session_->GetInputName(0));
      call.arrays.push_back(ensure_contiguous(input_feed));
    } else if (py::isinstance<py::dict>(input_feed)) {
      py::dict values = py::reinterpret_borrow<py::dict>(input_feed);
      if (static_cast<size_t>(values.size()) != count)
        throw std::runtime_error("Input dict size mismatch");
      for (size_t i = 0; i < count; ++i) {
        const auto &name = session_->GetInputName(i);
        if (!values.contains(py::str(name)))
          throw std::runtime_error("Missing input: " + name);
        call.names.push_back(name);
        call.arrays.push_back(ensure_contiguous(values[py::str(name)]));
      }
    } else if (py::isinstance<py::sequence>(input_feed)) {
      py::sequence values = py::reinterpret_borrow<py::sequence>(input_feed);
      if (static_cast<size_t>(values.size()) != count)
        throw std::runtime_error("Input list size mismatch");
      for (size_t i = 0; i < count; ++i) {
        call.names.push_back(session_->GetInputName(i));
        call.arrays.push_back(ensure_contiguous(values[i]));
      }
    } else {
      throw std::runtime_error(
          "input_feed must be a numpy array, list/tuple, or dict");
    }
    for (const auto &name : call.names)
      call.name_ptrs.push_back(name.c_str());
    for (auto &array : call.arrays)
      call.values.push_back(value_from_array(array));
    return call;
  }

  std::vector<std::string> output_names(py::handle value,
                                        bool expand_all = false) const {
    std::vector<std::string> result;
    if (value.is_none()) {
      if (expand_all)
        for (size_t i = 0; i < session_->GetOutputCount(); ++i)
          result.push_back(session_->GetOutputName(i));
      return result;
    }
    if (!py::isinstance<py::sequence>(value))
      throw std::runtime_error("output_names must be a sequence or None");
    for (py::handle name : value.cast<py::sequence>())
      result.push_back(py::cast<std::string>(name));
    return result;
  }

  py::list run(py::handle output_names_object, py::handle input_feed,
               py::handle run_options_object) {
    auto call = prepare_inputs(input_feed);
    auto names = output_names(output_names_object);
    std::vector<const char *> name_ptrs;
    for (const auto &name : names)
      name_ptrs.push_back(name.c_str());
    const bool all_outputs = names.empty() && output_names_object.is_none();
    const auto parsed_options = parse_run_options(run_options_object);
    std::vector<smr::Value> outputs;
    {
      py::gil_scoped_release release;
      outputs = session_->Run(
          parsed_options, call.name_ptrs.data(), call.values.data(),
          call.values.size(), all_outputs ? nullptr : name_ptrs.data(),
          names.size());
    }
    py::list result;
    for (auto &output : outputs)
      result.append(wrap_value(std::move(output))->numpy());
    return result;
  }

  struct AsyncState {
    std::vector<smr::Value> outputs;
    py::function callback;
    py::object user_data;
  };

  static void async_complete(void *opaque, smr::Value *, size_t count,
                             const char *error) {
    py::gil_scoped_acquire gil;
    // Keep the GIL alive while AsyncState is destroyed: it owns the Python
    // callback and user_data objects.
    std::unique_ptr<AsyncState> state(static_cast<AsyncState *>(opaque));
    try {
      if (error != nullptr && error[0] != '\0') {
        state->callback(py::none(), state->user_data, py::str(error));
        return;
      }
      py::list result;
      for (size_t i = 0; i < count; ++i)
        result.append(wrap_value(std::move(state->outputs[i]))->numpy());
      state->callback(result, state->user_data, py::str(""));
    } catch (const py::error_already_set &caught) {
      PyErr_WriteUnraisable(caught.value().ptr());
    }
  }

  void run_async(py::handle output_names_object, py::handle input_feed,
                 py::function callback, py::object user_data,
                 py::handle run_options_object) {
    auto call = prepare_inputs(input_feed);
    auto names = output_names(output_names_object, true);
    std::vector<const char *> name_ptrs;
    for (const auto &name : names)
      name_ptrs.push_back(name.c_str());
    auto state = std::make_unique<AsyncState>();
    state->outputs.resize(names.size());
    state->callback = std::move(callback);
    state->user_data = std::move(user_data);
    const auto parsed_options = parse_run_options(run_options_object);
    auto *raw_state = state.release();
    try {
      py::gil_scoped_release release;
      session_->RunAsync(parsed_options,
                         call.name_ptrs.data(), call.values.data(),
                         call.values.size(), name_ptrs.data(),
                         raw_state->outputs.data(), names.size(),
                         &PySession::async_complete, raw_state);
    } catch (...) {
      delete raw_state;
      throw;
    }
  }

  py::object run_pipeline(py::handle input_feed, size_t depth, bool reset) {
    auto call = prepare_inputs(input_feed);
    std::optional<std::vector<smr::Value>> outputs;
    {
      py::gil_scoped_release release;
      outputs = session_->RunPipeline(call.name_ptrs.data(), call.values.data(),
                                      call.values.size(), depth, reset);
    }
    if (!outputs)
      return py::none();
    py::list result;
    for (auto &output : *outputs)
      result.append(wrap_value(std::move(output))->numpy());
    return result;
  }

private:
  smr::Env env_;
  std::shared_ptr<smr::Session> session_;
};

std::shared_ptr<PyOrtValue>
make_session_value(PySession &session, py::array array,
                   const std::string &device_type, const std::string &name,
                   const std::string &io_kind, uint64_t memory_flags = 0) {
  py::array contiguous = ensure_contiguous(array);
  const auto kind = io_kind == "input" ? smr::IoKind::Input : smr::IoKind::Output;
  if (io_kind != "input" && io_kind != "output")
    throw std::runtime_error("io_kind must be 'input' or 'output'");
  if (device_type == "cpu")
    return std::make_shared<PyOrtValue>(value_from_array(contiguous), contiguous);
  if (device_type != "rknpu2")
    throw std::runtime_error("device_type must be 'cpu' or 'rknpu2'");
  auto value = session.session().CreateIoValue(
      name, kind, smr::MemoryInfo::CreateRknpu2(),
      smr::TensorElementType::Unknown, std::nullopt, memory_flags);
  const auto target_type = value.GetTensorTypeAndShapeInfo().GetElementType();
  if (!contiguous.dtype().is(element_to_dtype(target_type)))
    contiguous = py::array::ensure(
        contiguous.attr("astype")(element_to_dtype(target_type)),
        py::array::c_style);
  value.Update(contiguous.data(), static_cast<size_t>(contiguous.nbytes()),
               array_shape(contiguous), target_type);
  return std::make_shared<PyOrtValue>(std::move(value));
}

py::dict memory_info_dict(const PyOrtValue &value) {
  const auto info = value.value().GetMemoryInfo();
  py::dict result;
  result["device_type"] =
      info.device_type == smr::DeviceType::Cpu ? "cpu" : "rknpu2";
  result["device_id"] = info.device_id;
  if (info.device_type == smr::DeviceType::Rknpu2) {
    result["size"] = info.size;
    result["fd"] = info.fd;
    result["offset"] = info.offset;
    result["virt_addr"] = info.virt_addr;
    result["import_count"] = info.import_count;
    result["dmabuf_backed"] = info.dmabuf_backed;
    result["attr_size"] = info.attr_size;
    result["size_with_stride"] = info.size_with_stride;
    result["w_stride"] = info.w_stride;
    result["h_stride"] = info.h_stride;
    result["fmt"] = info.fmt;
    result["pass_through"] = info.pass_through;
  }
  return result;
}

} // namespace

PYBIND11_MODULE(_core, module) {
  module.doc() = "ORT-style Python bindings for the ztu RKNN runtime.";

  py::class_<smr::NodeArg>(module, "NodeArg")
      .def_property_readonly("name", [](const smr::NodeArg &value) {
        return value.name;
      })
      .def_property_readonly("shape", [](const smr::NodeArg &value) {
        return value.shape;
      })
      .def_property_readonly("type", [](const smr::NodeArg &value) {
        return value.type;
      });

  py::class_<smr::ModelMetadata>(module, "ModelMetadata")
      .def_property_readonly(
          "custom_metadata_map", [](const smr::ModelMetadata &value) {
            return value.custom_metadata_map;
          });

  py::class_<PyOrtValue, std::shared_ptr<PyOrtValue>>(module, "OrtValue")
      .def_static(
          "ortvalue_from_numpy",
          [](py::array array, const std::string &device_type, int device_id,
             py::object name, py::object session, py::object io_kind,
             const std::string &, bool) {
            if (device_id != 0)
              throw std::runtime_error("RKNN supports device_id 0 only");
            if (device_type == "cpu") {
              auto contiguous = ensure_contiguous(array);
              return std::make_shared<PyOrtValue>(value_from_array(contiguous),
                                                  contiguous);
            }
            if (session.is_none() || name.is_none() || io_kind.is_none())
              throw std::runtime_error(
                  "session, name, and io_kind are required for RKNN OrtValue");
            return make_session_value(*session.cast<PySession *>(), array,
                                      device_type, py::cast<std::string>(name),
                                      py::cast<std::string>(io_kind));
          },
          py::arg("arr"), py::arg("device_type") = "cpu",
          py::arg("device_id") = 0, py::kw_only(),
          py::arg("name") = py::none(), py::arg("session") = py::none(),
          py::arg("io_kind") = py::none(), py::arg("layout") = "onnx",
          py::arg("sync") = true)
      .def_static(
          "ortvalue_from_shape_and_type",
          [](py::handle shape_object, py::handle type_object,
             const std::string &device_type, int device_id, py::object name,
             py::object session, py::object io_kind, const std::string &,
             py::handle memory_flags) {
            if (device_id != 0)
              throw std::runtime_error("RKNN supports device_id 0 only");
            auto shape = parse_shape(shape_object);
            const auto type = dtype_like_to_element(type_object);
            if (device_type == "cpu") {
              smr::AllocatorWithDefaultOptions allocator;
              return wrap_value(smr::Value::CreateOwnedTensor(
                  shape.data(), shape.size(), type));
            }
            if (device_type != "rknpu2")
              throw std::runtime_error("device_type must be 'cpu' or 'rknpu2'");
            if (!session.is_none() && !name.is_none() && !io_kind.is_none()) {
              const std::string kind_text = py::cast<std::string>(io_kind);
              const auto kind = kind_text == "input" ? smr::IoKind::Input
                                                      : smr::IoKind::Output;
              return wrap_value(session.cast<PySession *>()->session().CreateIoValue(
                  py::cast<std::string>(name), kind,
                  smr::MemoryInfo::CreateRknpu2(), type, shape,
                  parse_memory_flags(memory_flags)));
            }
            std::vector<uint8_t> zeros(
                smr::TensorTypeAndShapeInfo(type, shape).GetElementCount() *
                element_to_dtype(type).itemsize());
            return wrap_value(smr::Value::CreateTensor(
                smr::MemoryInfo::CreateRknpu2(), zeros.data(), zeros.size(),
                shape.data(), shape.size(), type));
          },
          py::arg("shape"), py::arg("element_type"),
          py::arg("device_type") = "cpu", py::arg("device_id") = 0,
          py::kw_only(), py::arg("name") = py::none(),
          py::arg("session") = py::none(), py::arg("io_kind") = py::none(),
          py::arg("layout") = "native", py::arg("mem_flags") = "cacheable")
      .def_static(
          "ortvalue_from_dmabuf",
          [](int fd, py::handle shape_object, py::handle type_object,
             uint32_t size, int32_t offset, py::object virt_addr,
             py::object session, py::object name, py::object io_kind,
             const std::string &) {
            if (session.is_none() || name.is_none() || io_kind.is_none())
              throw std::runtime_error(
                  "session, name, and io_kind are required for dma-buf OrtValue");
            void *address = nullptr;
            if (!virt_addr.is_none())
              address = reinterpret_cast<void *>(static_cast<uintptr_t>(
                  py::cast<uint64_t>(virt_addr)));
            const std::string kind_text = py::cast<std::string>(io_kind);
            const auto kind = kind_text == "input" ? smr::IoKind::Input
                                                    : smr::IoKind::Output;
            return wrap_value(
                session.cast<PySession *>()->session().CreateIoValueFromDmaBuf(
                    py::cast<std::string>(name), kind, fd, size,
                    dtype_like_to_element(type_object), parse_shape(shape_object),
                    offset, address));
          },
          py::arg("fd"), py::arg("shape"), py::arg("element_type"),
          py::kw_only(), py::arg("size"), py::arg("offset") = 0,
          py::arg("virt_addr") = py::none(), py::arg("session") = py::none(),
          py::arg("name") = py::none(), py::arg("io_kind") = py::none(),
          py::arg("layout") = "native")
      .def("numpy", &PyOrtValue::numpy)
      .def("update_inplace", &PyOrtValue::update, py::arg("np_arr"))
      .def("data_ptr", [](const PyOrtValue &value) {
        return reinterpret_cast<uintptr_t>(value.value().GetTensorRawData());
      })
      .def("device_name", [](const PyOrtValue &value) {
        return value.value().GetMemoryInfo().device_type == smr::DeviceType::Cpu
                   ? "cpu"
                   : "rknpu2";
      })
      .def("device_type", [](const PyOrtValue &value) {
        return value.value().GetMemoryInfo().device_type == smr::DeviceType::Cpu
                   ? "cpu"
                   : "rknpu2";
      })
      .def("device_id", [](const PyOrtValue &value) {
        return value.value().GetMemoryInfo().device_id;
      })
      .def("shape", [](const PyOrtValue &value) {
        return value.value().GetTensorTypeAndShapeInfo().GetShape();
      })
      .def("element_type", [](const PyOrtValue &value) {
        return element_to_onnx_string(
            value.value().GetTensorTypeAndShapeInfo().GetElementType());
      })
      .def("is_tensor", [](const PyOrtValue &value) {
        return value.value().IsTensor();
      })
      .def("has_value", [](const PyOrtValue &value) {
        return value.value().HasValue();
      })
      .def("sync_to_device", [](PyOrtValue &value) {
        value.value().SyncToDevice();
      })
      .def("sync_from_device", [](PyOrtValue &value) {
        value.value().SyncFromDevice();
      })
      .def("memory_info", &memory_info_dict);

  py::class_<PyIOBinding, std::shared_ptr<PyIOBinding>>(module,
                                                        "SessionIOBinding")
      .def("bind_cpu_input", &PyIOBinding::bind_cpu_input, py::arg("name"),
           py::arg("arr_on_cpu"))
      .def("bind_input", [](PyIOBinding &, const std::string &,
                            const std::string &, int, py::handle, py::handle,
                            py::handle) {
        throw std::runtime_error(
            "bind_input with raw buffer_ptr is not supported; use an OrtValue "
            "or OrtValue.ortvalue_from_dmabuf");
      })
      .def("bind_ortvalue_input", &PyIOBinding::bind_value_input,
           py::arg("name"), py::arg("ortvalue"))
      .def("bind_ortvalue_output", &PyIOBinding::bind_value_output,
           py::arg("name"), py::arg("ortvalue"))
      .def("bind_output", [](PyIOBinding &binding, const std::string &name,
                             const std::string &device_type, int device_id,
                             py::handle element_type, py::handle shape,
                             py::handle buffer_ptr, const std::string &,
                             py::handle memory_flags) {
        if (!buffer_ptr.is_none())
          throw std::runtime_error(
              "bind_output with raw buffer_ptr is not implemented");
        std::optional<std::vector<int64_t>> parsed_shape;
        if (!shape.is_none())
          parsed_shape = parse_shape(shape);
        binding.bind_output(name, device_type, device_id,
                            dtype_like_to_element(element_type, true),
                            std::move(parsed_shape),
                            parse_memory_flags(memory_flags));
      }, py::arg("name"), py::arg("device_type") = "cpu",
           py::arg("device_id") = 0, py::arg("element_type") = py::none(),
           py::arg("shape") = py::none(), py::arg("buffer_ptr") = py::none(),
           py::kw_only(), py::arg("layout") = "native",
           py::arg("mem_flags") = "cacheable")
      .def("get_outputs", &PyIOBinding::outputs)
      .def("copy_outputs_to_cpu", &PyIOBinding::copy_outputs)
      .def("clear_binding_inputs", [](PyIOBinding &binding) {
        binding.binding().ClearBoundInputs();
      })
      .def("clear_binding_outputs", [](PyIOBinding &binding) {
        binding.binding().ClearBoundOutputs();
      })
      .def("synchronize_inputs", [](PyIOBinding &binding) {
        binding.binding().SynchronizeInputs();
      })
      .def("synchronize_outputs", [](PyIOBinding &binding) {
        binding.binding().SynchronizeOutputs();
      });
  module.attr("IOBinding") = module.attr("SessionIOBinding");

  py::class_<PySession>(module, "InferenceSession")
      .def(py::init([](py::handle path_or_bytes, py::handle,
                       py::handle, py::handle provider_options,
                       py::kwargs kwargs) {
        PySession *master = nullptr;
        if (kwargs.contains("use_weight_from") &&
            !kwargs["use_weight_from"].is_none())
          master = kwargs["use_weight_from"].cast<PySession *>();
        return std::make_unique<PySession>(path_or_bytes, provider_options,
                                           master);
      }), py::arg("path_or_bytes"), py::arg("sess_options") = py::none(),
           py::arg("providers") = py::none(),
           py::arg("provider_options") = py::none())
      .def("run", &PySession::run, py::arg("output_names") = py::none(),
           py::arg("input_feed"), py::arg("run_options") = py::none())
      .def("run_async", &PySession::run_async, py::arg("output_names"),
           py::arg("input_feed"), py::arg("callback"),
           py::arg("user_data") = py::none(),
           py::arg("run_options") = py::none())
      .def("run_pipeline", &PySession::run_pipeline, py::arg("input_feed"),
           py::arg("depth") = 3, py::arg("reset") = false)
      .def("io_binding", [](PySession &session) {
        return std::make_shared<PyIOBinding>(session.session());
      })
      .def("run_with_iobinding", [](PySession &session,
                                    PyIOBinding &binding, py::handle options) {
        auto parsed = parse_run_options(options);
        py::gil_scoped_release release;
        session.session().Run(parsed, binding.binding());
      }, py::arg("io_binding"), py::arg("run_options") = py::none())
      .def("get_inputs", [](const PySession &session) {
        return const_cast<PySession &>(session).session().GetInputs();
      })
      .def("get_outputs", [](const PySession &session) {
        return const_cast<PySession &>(session).session().GetOutputs();
      })
      .def("get_native_inputs", [](const PySession &session) {
        return const_cast<PySession &>(session).session().GetNativeInputs();
      })
      .def("get_native_outputs", [](const PySession &session) {
        return const_cast<PySession &>(session).session().GetNativeOutputs();
      })
      .def("get_modelmeta", [](const PySession &session) {
        return const_cast<PySession &>(session).session().GetModelMetadata();
      })
      .def_property_readonly("input_names", [](const PySession &session) {
        auto &native = const_cast<PySession &>(session).session();
        std::vector<std::string> result;
        for (size_t i = 0; i < native.GetInputCount(); ++i)
          result.push_back(native.GetInputName(i));
        return result;
      })
      .def_property_readonly("output_names", [](const PySession &session) {
        auto &native = const_cast<PySession &>(session).session();
        std::vector<std::string> result;
        for (size_t i = 0; i < native.GetOutputCount(); ++i)
          result.push_back(native.GetOutputName(i));
        return result;
      });
}
