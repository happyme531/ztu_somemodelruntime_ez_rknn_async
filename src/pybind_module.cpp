#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cctype>
#include <chrono>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "ez_rknn_async.hpp"

namespace py = pybind11;

namespace {
using ztu::rk::AsyncEzRknn;

struct ParsedInput {
  py::array array;
  rknn_tensor_type type;
  size_t bytes;
  const void *data;
};

struct PendingResult {
  std::future<AsyncEzRknn::InferenceResult> future;
};

struct NodeArgInfo {
  std::string name;
  std::vector<int64_t> shape;
  std::string type;
};

struct SessionConfig {
  std::string layout = "nchw_software";
  size_t max_queue_size = 3;
  int threads_per_core = 1;
  bool sequential_callbacks = true;
  std::vector<uint64_t> schedule = {0};
  bool enable_pacing = false;
  bool disable_dup_context = false;
  std::vector<std::string> custom_op_paths;
  bool custom_op_default_path = false;
};

template <typename T> std::shared_ptr<T> make_py_shared(T obj) {
  return std::shared_ptr<T>(new T(std::move(obj)), [](T *p) {
    py::gil_scoped_acquire gil;
    delete p;
  });
}

rknn_tensor_type dtype_to_rknn(const py::dtype &dtype) {
  if (dtype.is(py::dtype::of<float>())) {
    return RKNN_TENSOR_FLOAT32;
  }
  if (dtype.is(py::dtype("float16"))) {
    return RKNN_TENSOR_FLOAT16;
  }
  if (dtype.is(py::dtype::of<int8_t>())) {
    return RKNN_TENSOR_INT8;
  }
  if (dtype.is(py::dtype::of<uint8_t>())) {
    return RKNN_TENSOR_UINT8;
  }
  if (dtype.is(py::dtype::of<int16_t>())) {
    return RKNN_TENSOR_INT16;
  }
  if (dtype.is(py::dtype::of<uint16_t>())) {
    return RKNN_TENSOR_UINT16;
  }
  if (dtype.is(py::dtype::of<int32_t>())) {
    return RKNN_TENSOR_INT32;
  }
  if (dtype.is(py::dtype::of<uint32_t>())) {
    return RKNN_TENSOR_UINT32;
  }
  if (dtype.is(py::dtype::of<int64_t>())) {
    return RKNN_TENSOR_INT64;
  }
  if (dtype.is(py::dtype::of<bool>())) {
    return RKNN_TENSOR_BOOL;
  }
  throw std::runtime_error("Unsupported numpy dtype for RKNN input");
}

size_t rknn_type_size(rknn_tensor_type type) {
  switch (type) {
  case RKNN_TENSOR_FLOAT32:
    return sizeof(float);
  case RKNN_TENSOR_FLOAT16:
    return sizeof(uint16_t);
  case RKNN_TENSOR_INT8:
    return sizeof(int8_t);
  case RKNN_TENSOR_UINT8:
    return sizeof(uint8_t);
  case RKNN_TENSOR_INT16:
    return sizeof(int16_t);
  case RKNN_TENSOR_UINT16:
    return sizeof(uint16_t);
  case RKNN_TENSOR_INT32:
    return sizeof(int32_t);
  case RKNN_TENSOR_UINT32:
    return sizeof(uint32_t);
  case RKNN_TENSOR_INT64:
    return sizeof(int64_t);
  case RKNN_TENSOR_BOOL:
    return sizeof(bool);
  default:
    throw std::runtime_error("Unsupported rknn tensor type");
  }
}

std::string rknn_type_to_onnx_str(rknn_tensor_type type) {
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

std::string trim_string(const std::string &input) {
  size_t start = 0;
  while (start < input.size() &&
         std::isspace(static_cast<unsigned char>(input[start]))) {
    ++start;
  }
  size_t end = input.size();
  while (end > start &&
         std::isspace(static_cast<unsigned char>(input[end - 1]))) {
    --end;
  }
  return input.substr(start, end - start);
}

int64_t parse_int_like(py::handle value, const char *name) {
  if (py::isinstance<py::int_>(value)) {
    return py::cast<int64_t>(value);
  }
  if (py::isinstance<py::str>(value)) {
    std::string text = trim_string(py::cast<std::string>(value));
    size_t idx = 0;
    long long parsed = 0;
    try {
      parsed = std::stoll(text, &idx);
    } catch (...) {
      throw std::runtime_error(std::string(name) + " must be an integer");
    }
    if (idx != text.size()) {
      throw std::runtime_error(std::string(name) + " must be an integer");
    }
    return static_cast<int64_t>(parsed);
  }
  throw std::runtime_error(std::string(name) + " must be an integer");
}

bool parse_bool_like(py::handle value, const char *name) {
  if (py::isinstance<py::bool_>(value)) {
    return py::cast<bool>(value);
  }
  if (py::isinstance<py::int_>(value)) {
    return py::cast<int64_t>(value) != 0;
  }
  if (py::isinstance<py::str>(value)) {
    std::string text = trim_string(py::cast<std::string>(value));
    for (auto &c : text) {
      c = static_cast<char>(::tolower(c));
    }
    if (text == "1" || text == "true" || text == "on" || text == "yes") {
      return true;
    }
    if (text == "0" || text == "false" || text == "off" || text == "no") {
      return false;
    }
  }
  throw std::runtime_error(std::string(name) + " must be a boolean");
}

std::vector<uint64_t> parse_schedule_like(py::handle value) {
  std::vector<uint64_t> schedule;
  if (py::isinstance<py::int_>(value) || py::isinstance<py::str>(value)) {
    if (py::isinstance<py::int_>(value)) {
      int64_t core = parse_int_like(value, "schedule");
      if (core < 0) {
        throw std::runtime_error("schedule values must be >= 0");
      }
      schedule.push_back(static_cast<uint64_t>(core));
    } else {
      std::string text = py::cast<std::string>(value);
      for (auto &c : text) {
        if (c == '[' || c == ']' || c == ',') {
          c = ' ';
        }
      }
      std::stringstream ss(text);
      std::string token;
      while (ss >> token) {
        size_t idx = 0;
        long long parsed = 0;
        try {
          parsed = std::stoll(token, &idx);
        } catch (...) {
          throw std::runtime_error("schedule must contain integers");
        }
        if (idx != token.size()) {
          throw std::runtime_error("schedule must contain integers");
        }
        if (parsed < 0) {
          throw std::runtime_error("schedule values must be >= 0");
        }
        schedule.push_back(static_cast<uint64_t>(parsed));
      }
    }
  } else if (py::isinstance<py::sequence>(value)) {
    py::sequence seq = value.cast<py::sequence>();
    schedule.reserve(seq.size());
    for (auto item : seq) {
      int64_t core = parse_int_like(item, "schedule");
      if (core < 0) {
        throw std::runtime_error("schedule values must be >= 0");
      }
      schedule.push_back(static_cast<uint64_t>(core));
    }
  } else {
    throw std::runtime_error(
        "schedule must be an integer, string, or sequence");
  }
  if (schedule.empty()) {
    throw std::runtime_error("schedule must not be empty");
  }
  return schedule;
}

std::string parse_path_like(py::handle value, const char *name) {
  try {
    py::object path_obj = py::module_::import("os").attr("fspath")(value);
    return py::cast<std::string>(path_obj);
  } catch (...) {
    throw std::runtime_error(std::string(name) +
                             " must be a path-like object or string");
  }
}

std::vector<std::string> parse_paths_like(py::handle value, const char *name) {
  if (py::isinstance<py::str>(value) || py::hasattr(value, "__fspath__")) {
    return {parse_path_like(value, name)};
  }
  if (py::isinstance<py::sequence>(value)) {
    py::sequence seq = value.cast<py::sequence>();
    std::vector<std::string> paths;
    paths.reserve(seq.size());
    for (auto item : seq) {
      paths.push_back(parse_path_like(item, name));
    }
    return paths;
  }
  throw std::runtime_error(std::string(name) +
                           " must be a path-like object or sequence of "
                           "path-like objects");
}

py::dict extract_provider_options(const py::object &provider_options_obj) {
  if (provider_options_obj.is_none()) {
    return py::dict();
  }
  if (py::isinstance<py::dict>(provider_options_obj)) {
    return provider_options_obj.cast<py::dict>();
  }
  if (py::isinstance<py::sequence>(provider_options_obj)) {
    py::sequence seq = provider_options_obj.cast<py::sequence>();
    if (seq.size() == 0) {
      return py::dict();
    }
    py::object first = seq[0];
    if (py::isinstance<py::dict>(first)) {
      return first.cast<py::dict>();
    }
  }
  throw std::runtime_error("provider_options must be a dict or a sequence "
                           "whose first element is a dict");
}

SessionConfig parse_session_config(const py::object &provider_options_obj) {
  SessionConfig config;
  py::dict opts = extract_provider_options(provider_options_obj);
  for (auto item : opts) {
    std::string key = py::cast<std::string>(item.first);
    if (key == "layout") {
      config.layout = py::cast<std::string>(item.second);
      continue;
    }
    if (key == "max_queue_size") {
      int64_t value = parse_int_like(item.second, "max_queue_size");
      if (value <= 0) {
        throw std::runtime_error("max_queue_size must be > 0");
      }
      config.max_queue_size = static_cast<size_t>(value);
      continue;
    }
    if (key == "threads_per_core") {
      int64_t value = parse_int_like(item.second, "threads_per_core");
      if (value <= 0) {
        throw std::runtime_error("threads_per_core must be > 0");
      }
      config.threads_per_core = static_cast<int>(value);
      continue;
    }
    if (key == "sequential_callbacks") {
      config.sequential_callbacks = parse_bool_like(item.second, key.c_str());
      continue;
    }
    if (key == "schedule") {
      config.schedule = parse_schedule_like(item.second);
      continue;
    }
    if (key == "enable_pacing") {
      config.enable_pacing = parse_bool_like(item.second, key.c_str());
      continue;
    }
    if (key == "disable_dup_context") {
      config.disable_dup_context = parse_bool_like(item.second, key.c_str());
      continue;
    }
    if (key == "custom_op_path" || key == "custom_op_paths") {
      auto parsed_paths = parse_paths_like(item.second, key.c_str());
      config.custom_op_paths.insert(config.custom_op_paths.end(),
                                    parsed_paths.begin(), parsed_paths.end());
      continue;
    }
    if (key == "custom_op_default_path" ||
        key == "load_custom_ops_from_default_path") {
      config.custom_op_default_path = parse_bool_like(item.second, key.c_str());
      continue;
    }
    throw std::runtime_error(
        "Unknown provider_options key: " + key +
        ". Supported keys: layout, max_queue_size, threads_per_core, "
        "sequential_callbacks, schedule, enable_pacing, disable_dup_context, "
        "custom_op_path, custom_op_paths, custom_op_default_path, "
        "load_custom_ops_from_default_path");
  }
  return config;
}

std::string resolve_model_path(const py::object &path_or_bytes) {
  if (py::isinstance<py::bytes>(path_or_bytes)) {
    throw std::runtime_error("bytes model content is not supported yet; please "
                             "pass a filesystem path");
  }
  py::object os_path = py::module_::import("os").attr("fspath")(path_or_bytes);
  return py::cast<std::string>(os_path);
}

ParsedInput parse_array(py::array array, size_t expected_elems,
                        bool allow_batch) {
  if (!array) {
    throw std::runtime_error("Input must be a numpy array");
  }
  auto dtype = array.dtype();
  rknn_tensor_type type = dtype_to_rknn(dtype);
  size_t elems = static_cast<size_t>(array.size());
  if (!allow_batch && expected_elems != elems) {
    throw std::runtime_error("Input element count mismatch: expected " +
                             std::to_string(expected_elems) + ", got " +
                             std::to_string(elems));
  }
  return ParsedInput{array, type, static_cast<size_t>(array.nbytes()),
                     array.data()};
}

py::array make_output_array(const std::shared_ptr<float[]> &data,
                            const rknn_tensor_attr &attr) {
  int ndim = static_cast<int>(attr.n_dims);
  std::vector<ssize_t> shape;
  shape.reserve(ndim);
  for (int i = 0; i < ndim; ++i) {
    shape.push_back(static_cast<ssize_t>(attr.dims[i]));
  }
  std::vector<ssize_t> strides;
  strides.resize(ndim);
  ssize_t stride = static_cast<ssize_t>(sizeof(float));
  for (int i = ndim - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape[i];
  }
  auto holder = new std::shared_ptr<float[]>(data);
  py::capsule capsule(holder, [](void *p) {
    auto *ptr = static_cast<std::shared_ptr<float[]> *>(p);
    delete ptr;
  });
  return py::array(py::buffer_info(data.get(), sizeof(float),
                                   py::format_descriptor<float>::format(), ndim,
                                   shape, strides),
                   capsule);
}

std::vector<size_t>
resolve_output_indices(py::object output_names_obj,
                       const std::vector<std::string> &all_names) {
  std::vector<size_t> indices;
  if (output_names_obj.is_none()) {
    indices.resize(all_names.size());
    for (size_t i = 0; i < all_names.size(); ++i) {
      indices[i] = i;
    }
    return indices;
  }
  if (!py::isinstance<py::sequence>(output_names_obj)) {
    throw std::runtime_error("output_names must be a sequence or None");
  }
  py::sequence seq = output_names_obj.cast<py::sequence>();
  indices.reserve(seq.size());
  for (auto item : seq) {
    std::string name = py::cast<std::string>(item);
    bool found = false;
    for (size_t i = 0; i < all_names.size(); ++i) {
      if (all_names[i] == name) {
        indices.push_back(i);
        found = true;
        break;
      }
    }
    if (!found) {
      throw std::runtime_error("Unknown output name: " + name);
    }
  }
  return indices;
}

py::object outputs_to_python(const AsyncEzRknn::InferenceResult &outputs,
                             const std::vector<rknn_tensor_attr> &attrs) {
  py::list result;
  for (size_t i = 0; i < outputs.size(); ++i) {
    result.append(make_output_array(outputs[i], attrs[i]));
  }
  return result;
}

py::list build_outputs_by_indices(const AsyncEzRknn::InferenceResult &outputs,
                                  const std::vector<rknn_tensor_attr> &attrs,
                                  const std::vector<size_t> &indices) {
  py::list result;
  for (size_t idx : indices) {
    if (idx >= outputs.size() || idx >= attrs.size()) {
      throw std::runtime_error("Output index out of range");
    }
    result.append(make_output_array(outputs[idx], attrs[idx]));
  }
  return result;
}

bool parse_dispatch_batch_flag(const py::object &run_options_obj) {
  constexpr const char *kDispatchKey = "ztu_modelrt_dispatch_batch";
  if (run_options_obj.is_none()) {
    return false;
  }
  if (py::isinstance<py::dict>(run_options_obj)) {
    py::dict opts = run_options_obj.cast<py::dict>();
    py::object key = py::str(kDispatchKey);
    if (!opts.contains(key)) {
      return false;
    }
    return parse_bool_like(opts[key], kDispatchKey);
  }
  if (py::hasattr(run_options_obj, "get_run_config_entry")) {
    try {
      py::object value =
          run_options_obj.attr("get_run_config_entry")(kDispatchKey);
      return parse_bool_like(value, kDispatchKey);
    } catch (const py::error_already_set &) {
      PyErr_Clear();
      return false;
    }
  }
  return false;
}

void emit_python_user_warning(const std::string &message) {
  py::module_ warnings = py::module_::import("warnings");
  py::object user_warning = py::module_::import("builtins").attr("UserWarning");
  warnings.attr("warn")(py::str(message), user_warning, 2);
}

class InferenceSession {
public:
  ~InferenceSession() {
    // Avoid deadlock: Async worker/callback threads may need the GIL while
    // shutting down callbacks that hold Python objects.
    py::gil_scoped_release release;
    rknn_.reset();
  }

  InferenceSession(const std::string &model_path, const SessionConfig &config)
      : pipeline_depth_(0), pipeline_initialized_(false),
        nchw_software_(false) {
    const bool has_custom_op_request =
        !config.custom_op_paths.empty() || config.custom_op_default_path;
    const bool disable_dup_context =
        config.disable_dup_context || has_custom_op_request;

    if (has_custom_op_request) {
      emit_python_user_warning(
          "Custom op loading requested; dup_context is disabled for this "
          "session to avoid known RKNN stability issues.");
    }

    AsyncEzRknn::Layout layout_enum = AsyncEzRknn::Layout::ORIGINAL;
    std::string layout_lower = config.layout;
    for (auto &c : layout_lower) {
      c = static_cast<char>(::tolower(c));
    }
    if (layout_lower == "nchw" || layout_lower == "original") {
      layout_enum = AsyncEzRknn::Layout::NCHW;
    } else if (layout_lower == "nchw_software" ||
               layout_lower == "original_software") {
      layout_enum = AsyncEzRknn::Layout::NHWC;
      nchw_software_ = true;
    } else if (layout_lower == "nhwc") {
      layout_enum = AsyncEzRknn::Layout::NHWC;
    } else if (layout_lower == "any") {
      layout_enum = AsyncEzRknn::Layout::ANY;
    } else {
      throw std::runtime_error("Unknown layout: " + config.layout);
    }

    rknn_ = std::make_unique<AsyncEzRknn>(
        model_path, layout_enum, config.max_queue_size, config.threads_per_core,
        config.sequential_callbacks, config.schedule, config.enable_pacing,
        disable_dup_context);
    if (rknn_->sdk_version_warning().has_value()) {
      emit_python_user_warning(rknn_->sdk_version_warning().value());
    }
    for (const auto &custom_op_path : config.custom_op_paths) {
      rknn_->load_custom_op(custom_op_path);
    }
    if (config.custom_op_default_path) {
      rknn_->load_custom_ops_from_default_path();
    }

    input_attrs_ = rknn_->input_attrs;
    output_attrs_ = rknn_->output_attrs;

    if (nchw_software_) {
      for (auto &attr : input_attrs_) {
        if (attr.n_dims == 4) {
          auto n = attr.dims[0];
          auto h = attr.dims[1];
          auto w = attr.dims[2];
          auto c = attr.dims[3];
          attr.dims[0] = n;
          attr.dims[1] = c;
          attr.dims[2] = h;
          attr.dims[3] = w;
          if (attr.fmt == RKNN_TENSOR_NHWC) {
            attr.fmt = RKNN_TENSOR_NCHW;
          }
        }
      }
    }

    input_names_.reserve(input_attrs_.size());
    for (size_t i = 0; i < input_attrs_.size(); ++i) {
      const auto &attr = input_attrs_[i];
      std::string name =
          attr.name[0] ? std::string(attr.name) : "input_" + std::to_string(i);
      input_names_.push_back(name);
    }

    output_names_.reserve(output_attrs_.size());
    for (size_t i = 0; i < output_attrs_.size(); ++i) {
      const auto &attr = output_attrs_[i];
      std::string name =
          attr.name[0] ? std::string(attr.name) : "output_" + std::to_string(i);
      output_names_.push_back(name);
    }
  }

  py::list run(py::object output_names_obj, py::object input_feed,
               py::object run_options) {
    bool dispatch_batch = parse_dispatch_batch_flag(run_options);
    auto output_indices =
        resolve_output_indices(output_names_obj, output_names_);
    auto inputs = parse_input_feed(input_feed, dispatch_batch);
    if (dispatch_batch) {
      return run_batch(std::move(inputs), output_indices);
    }
    auto outputs = run_sync(std::move(inputs));
    return build_outputs(outputs, output_indices);
  }

  py::none run_async(py::object output_names_obj, py::object input_feed,
                     py::object callback, py::object user_data,
                     py::object run_options) {
    if (callback.is_none() || !PyCallable_Check(callback.ptr())) {
      throw std::runtime_error("callback must be callable");
    }
    if (parse_dispatch_batch_flag(run_options)) {
      throw std::runtime_error(
          "run_async does not support ztu_modelrt_dispatch_batch=True");
    }

    auto output_indices =
        resolve_output_indices(output_names_obj, output_names_);
    auto inputs = parse_input_feed(input_feed, false);
    auto cb_shared = make_py_shared(callback.cast<py::function>());
    auto user_data_shared = make_py_shared(user_data);
    auto attrs_shared =
        std::make_shared<std::vector<rknn_tensor_attr>>(output_attrs_);
    auto indices_shared =
        std::make_shared<std::vector<size_t>>(std::move(output_indices));

    submit_task_async_blocking(
        std::move(inputs),
        [cb_shared, user_data_shared, attrs_shared,
         indices_shared](size_t, AsyncEzRknn::InferenceResult outputs) {
          py::gil_scoped_acquire gil;
          try {
            if (outputs.empty()) {
              (*cb_shared)(py::none(), *user_data_shared,
                           py::str("Inference failed"));
              return;
            }
            py::list result = build_outputs_by_indices(outputs, *attrs_shared,
                                                       *indices_shared);
            (*cb_shared)(result, *user_data_shared, py::str(""));
          } catch (const py::error_already_set &e) {
            PyErr_WriteUnraisable(e.value().ptr());
          } catch (const std::exception &e) {
            try {
              (*cb_shared)(py::none(), *user_data_shared, py::str(e.what()));
            } catch (const py::error_already_set &e2) {
              PyErr_WriteUnraisable(e2.value().ptr());
            }
          }
        });
    return py::none();
  }

  py::object run_pipeline(py::object input_feed, size_t depth, bool reset) {
    if (depth == 0) {
      throw std::runtime_error("pipeline depth must be > 0");
    }

    auto inputs = parse_input_feed(input_feed, false);
    PendingResult pending;
    pending.future = submit_task_blocking(std::move(inputs));

    std::unique_lock<std::mutex> lock(pipeline_mutex_);
    if (!pipeline_initialized_ || reset || depth != pipeline_depth_) {
      pipeline_depth_ = depth;
      pipeline_initialized_ = true;
      while (!pipeline_queue_.empty()) {
        pipeline_queue_.pop();
      }
    }

    pipeline_queue_.push(std::move(pending));
    if (pipeline_queue_.size() <= pipeline_depth_) {
      return py::none();
    }

    PendingResult ready = std::move(pipeline_queue_.front());
    pipeline_queue_.pop();
    lock.unlock();

    AsyncEzRknn::InferenceResult outputs;
    {
      py::gil_scoped_release release;
      outputs = ready.future.get();
    }
    if (outputs.empty()) {
      throw std::runtime_error("Inference failed");
    }
    return outputs_to_python(outputs, output_attrs_);
  }

  std::vector<std::string> input_names() const { return input_names_; }
  std::vector<std::string> output_names() const { return output_names_; }
  std::vector<NodeArgInfo> get_inputs() const {
    return build_node_args(input_attrs_, input_names_);
  }
  std::vector<NodeArgInfo> get_outputs() const {
    return build_node_args(output_attrs_, output_names_);
  }

private:
  std::vector<ParsedInput> parse_input_feed(const py::object &input_feed,
                                            bool allow_batch) {
    size_t expected_inputs = input_attrs_.size();
    if (py::isinstance<py::array>(input_feed)) {
      if (expected_inputs != 1) {
        throw std::runtime_error("Model expects " +
                                 std::to_string(expected_inputs) +
                                 " inputs, but a single array was given");
      }
      py::array arr = ensure_array(input_feed, 0);
      return {parse_array(arr, input_attrs_[0].n_elems, allow_batch)};
    }

    if (py::isinstance<py::dict>(input_feed)) {
      py::dict d = input_feed.cast<py::dict>();
      if (d.size() != expected_inputs) {
        throw std::runtime_error("Input dict size mismatch: expected " +
                                 std::to_string(expected_inputs) + ", got " +
                                 std::to_string(static_cast<size_t>(d.size())));
      }
      std::vector<ParsedInput> inputs;
      inputs.reserve(expected_inputs);
      for (size_t i = 0; i < expected_inputs; ++i) {
        py::object key = py::str(input_names_[i]);
        if (!d.contains(key)) {
          throw std::runtime_error("Missing input: " + input_names_[i]);
        }
        py::array arr = ensure_array(d[key], i);
        inputs.push_back(
            parse_array(arr, input_attrs_[i].n_elems, allow_batch));
      }
      return inputs;
    }

    if (py::isinstance<py::sequence>(input_feed)) {
      py::sequence seq = input_feed.cast<py::sequence>();
      if (static_cast<size_t>(seq.size()) != expected_inputs) {
        throw std::runtime_error(
            "Input list size mismatch: expected " +
            std::to_string(expected_inputs) + ", got " +
            std::to_string(static_cast<size_t>(seq.size())));
      }
      std::vector<ParsedInput> inputs;
      inputs.reserve(expected_inputs);
      for (size_t i = 0; i < expected_inputs; ++i) {
        py::array arr = ensure_array(seq[i], i);
        inputs.push_back(
            parse_array(arr, input_attrs_[i].n_elems, allow_batch));
      }
      return inputs;
    }

    throw std::runtime_error(
        "input_feed must be a numpy array, list/tuple, or dict");
  }

  py::array ensure_array(py::handle obj, size_t input_index) {
    py::array array = py::array::ensure(obj, py::array::c_style);
    if (!array) {
      throw std::runtime_error("Input must be a numpy array");
    }
    if (nchw_software_ && input_attrs_[input_index].n_dims == 4) {
      return transpose_nchw_to_nhwc(array);
    }
    return array;
  }

  py::array transpose_nchw_to_nhwc(const py::array &array) {
    if (array.ndim() != 4) {
      return array;
    }
    py::object transposed = array.attr("transpose")(py::make_tuple(0, 2, 3, 1));
    py::array contiguous = py::array::ensure(transposed, py::array::c_style);
    if (!contiguous) {
      throw std::runtime_error("Failed to transpose input to NHWC");
    }
    return contiguous;
  }

  std::vector<AsyncEzRknn::InputView>
  to_input_views(const std::vector<ParsedInput> &inputs) {
    std::vector<AsyncEzRknn::InputView> views;
    views.reserve(inputs.size());
    for (const auto &in : inputs) {
      views.push_back(AsyncEzRknn::InputView{in.data, in.bytes, in.type});
    }
    return views;
  }

  void submit_task_async_blocking(std::vector<ParsedInput> inputs,
                                  AsyncEzRknn::InferenceCallback callback) {
    auto views = to_input_views(inputs);
    while (true) {
      auto submitted = rknn_->asyncInferenceDyn(callback, views);
      if (submitted.has_value()) {
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  std::future<AsyncEzRknn::InferenceResult>
  submit_task_blocking(std::vector<ParsedInput> inputs) {
    auto promise =
        std::make_shared<std::promise<AsyncEzRknn::InferenceResult>>();
    auto future = promise->get_future();
    auto views = to_input_views(inputs);
    while (true) {
      auto submitted = rknn_->asyncInferenceDyn(
          [promise](size_t, AsyncEzRknn::InferenceResult outputs) {
            try {
              promise->set_value(std::move(outputs));
            } catch (...) {
              try {
                promise->set_exception(std::current_exception());
              } catch (...) {
              }
            }
          },
          views);
      if (submitted.has_value()) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return future;
  }

  AsyncEzRknn::InferenceResult run_sync(std::vector<ParsedInput> inputs) {
    auto future = submit_task_blocking(std::move(inputs));
    py::gil_scoped_release release;
    auto outputs = future.get();
    if (outputs.empty()) {
      throw std::runtime_error("Inference failed");
    }
    return outputs;
  }

  py::list build_outputs(const AsyncEzRknn::InferenceResult &outputs,
                         const std::vector<size_t> &indices) {
    py::list result;
    for (size_t idx : indices) {
      if (idx >= outputs.size()) {
        throw std::runtime_error("Output index out of range");
      }
      result.append(make_output_array(outputs[idx], output_attrs_[idx]));
    }
    return result;
  }

  py::list run_batch(std::vector<ParsedInput> inputs,
                     const std::vector<size_t> &indices) {
    if (inputs.empty()) {
      throw std::runtime_error("No inputs provided");
    }
    size_t batch = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
      auto info = inputs[i].array.request();
      if (info.ndim < 1) {
        throw std::runtime_error("Batch input must have ndim >= 1");
      }
      size_t this_batch = static_cast<size_t>(info.shape[0]);
      if (i == 0) {
        batch = this_batch;
      } else if (this_batch != batch) {
        throw std::runtime_error("Batch size mismatch across inputs");
      }

      const auto &attr = input_attrs_[i];
      if (attr.n_dims == 0 || attr.dims[0] != 1) {
        throw std::runtime_error(
            "Batch mode requires model input batch dimension == 1");
      }

      size_t per_sample_elems = attr.n_elems / attr.dims[0];
      size_t expected_total_elems = per_sample_elems * batch;
      size_t actual_total_elems = static_cast<size_t>(info.size);
      if (actual_total_elems != expected_total_elems) {
        throw std::runtime_error("Batch input size mismatch at input index " +
                                 std::to_string(i));
      }
    }

    std::vector<std::future<AsyncEzRknn::InferenceResult>> futures;
    futures.reserve(batch);

    for (size_t b = 0; b < batch; ++b) {
      std::vector<ParsedInput> per_inputs;
      per_inputs.reserve(inputs.size());
      for (size_t i = 0; i < inputs.size(); ++i) {
        auto info = inputs[i].array.request();
        size_t stride0 = static_cast<size_t>(info.strides[0]);
        const char *base = static_cast<const char *>(info.ptr);
        const char *ptr = base + stride0 * b;

        const auto &attr = input_attrs_[i];
        size_t per_sample_elems = attr.n_elems / attr.dims[0];
        size_t bytes = per_sample_elems * rknn_type_size(inputs[i].type);

        ParsedInput one{inputs[i].array, inputs[i].type, bytes, ptr};
        per_inputs.push_back(one);
      }
      futures.push_back(submit_task_blocking(std::move(per_inputs)));
    }

    std::vector<AsyncEzRknn::InferenceResult> batch_outputs;
    batch_outputs.resize(batch);
    {
      py::gil_scoped_release release;
      for (size_t b = 0; b < batch; ++b) {
        batch_outputs[b] = futures[b].get();
        if (batch_outputs[b].empty()) {
          throw std::runtime_error("Inference failed in batch mode");
        }
      }
    }

    py::list result;
    for (size_t out_idx : indices) {
      const auto &attr = output_attrs_[out_idx];
      size_t per_sample_elems = attr.n_elems;
      std::vector<ssize_t> shape;
      if (attr.n_dims == 0) {
        shape = {static_cast<ssize_t>(batch)};
      } else {
        shape.reserve(attr.n_dims);
        for (uint32_t d = 0; d < attr.n_dims; ++d) {
          shape.push_back(static_cast<ssize_t>(d == 0 ? batch : attr.dims[d]));
        }
      }

      py::array_t<float> out_array(shape);
      float *dst = static_cast<float *>(out_array.request().ptr);

      for (size_t b = 0; b < batch; ++b) {
        const auto &out = batch_outputs[b][out_idx];
        std::memcpy(dst + b * per_sample_elems, out.get(),
                    per_sample_elems * sizeof(float));
      }
      result.append(out_array);
    }

    return result;
  }

  static std::vector<NodeArgInfo>
  build_node_args(const std::vector<rknn_tensor_attr> &attrs,
                  const std::vector<std::string> &names) {
    std::vector<NodeArgInfo> infos;
    infos.reserve(attrs.size());
    for (size_t i = 0; i < attrs.size(); ++i) {
      NodeArgInfo info;
      info.name = names[i];
      info.shape.reserve(attrs[i].n_dims);
      for (uint32_t d = 0; d < attrs[i].n_dims; ++d) {
        info.shape.push_back(static_cast<int64_t>(attrs[i].dims[d]));
      }
      info.type = rknn_type_to_onnx_str(attrs[i].type);
      infos.push_back(std::move(info));
    }
    return infos;
  }

  std::unique_ptr<AsyncEzRknn> rknn_;
  std::vector<rknn_tensor_attr> input_attrs_;
  std::vector<rknn_tensor_attr> output_attrs_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  bool nchw_software_;

  size_t pipeline_depth_;
  bool pipeline_initialized_;
  std::queue<PendingResult> pipeline_queue_;
  std::mutex pipeline_mutex_;
};

} // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "Python bindings for ztu_somemodelruntime_ez_rknn_async.";

  py::class_<NodeArgInfo>(m, "NodeArg")
      .def_property_readonly("name",
                             [](const NodeArgInfo &info) { return info.name; })
      .def_property_readonly("shape",
                             [](const NodeArgInfo &info) { return info.shape; })
      .def_property_readonly(
          "type", [](const NodeArgInfo &info) { return info.type; },
          "Type string like 'tensor(float32)'.");

  py::class_<InferenceSession>(m, "InferenceSession")
      .def(py::init([](py::object path_or_bytes, py::object sess_options,
                       py::object providers, py::object provider_options,
                       py::kwargs kwargs) {
             (void)sess_options;
             (void)providers;
             (void)kwargs;
             SessionConfig config = parse_session_config(provider_options);
             return std::make_unique<InferenceSession>(
                 resolve_model_path(path_or_bytes), config);
           }),
           py::arg("path_or_bytes"), py::arg("sess_options") = py::none(),
           py::arg("providers") = py::none(),
           py::arg("provider_options") = py::none(),
           "Create an inference session.\n\n"
           "ORT-style constructor. provider_options carries RKNN runtime "
           "options, including custom op loading settings.")
      .def("run", &InferenceSession::run, py::arg("output_names") = py::none(),
           py::arg("input_feed"), py::arg("run_options") = py::none(),
           "Run inference.\n\n"
           "output_names: list of output names or None for all.\n"
           "input_feed: dict(name->ndarray), list/tuple of ndarrays, or a "
           "single ndarray.\n"
           "run_options: supports key 'ztu_modelrt_dispatch_batch' (bool).")
      .def("run_async", &InferenceSession::run_async, py::arg("output_names"),
           py::arg("input_feed"), py::arg("callback"),
           py::arg("user_data") = py::none(),
           py::arg("run_options") = py::none(),
           "Run inference asynchronously.\n\n"
           "ORT-style callback signature: callback(results, user_data, err).\n"
           "Returns None.")
      .def("run_pipeline", &InferenceSession::run_pipeline,
           py::arg("input_feed"), py::arg("depth") = 3,
           py::arg("reset") = false,
           "Pipeline mode. Returns None until the pipeline is filled, then "
           "returns\n"
           "the oldest pending result. Use reset=True to clear the pipeline.")
      .def("get_inputs", &InferenceSession::get_inputs,
           "Return input NodeArg list (onnxruntime-style).")
      .def("get_outputs", &InferenceSession::get_outputs,
           "Return output NodeArg list (onnxruntime-style).")
      .def_property_readonly("input_names", &InferenceSession::input_names)
      .def_property_readonly("output_names", &InferenceSession::output_names);

  m.attr("__version__") = "0.3.0";
}
