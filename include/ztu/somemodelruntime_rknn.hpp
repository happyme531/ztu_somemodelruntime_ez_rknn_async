#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#if defined(_WIN32)
#if defined(ZTU_SOMEMODELRUNTIME_RKNN_BUILDING_LIBRARY)
#define ZTU_SOMEMODELRUNTIME_RKNN_API __declspec(dllexport)
#else
#define ZTU_SOMEMODELRUNTIME_RKNN_API __declspec(dllimport)
#endif
#elif defined(__GNUC__)
#define ZTU_SOMEMODELRUNTIME_RKNN_API __attribute__((visibility("default")))
#else
#define ZTU_SOMEMODELRUNTIME_RKNN_API
#endif

namespace ztu::rk {
class RknnOrtValue;
class RknnIoBinding;
class ZeroCopyEzRknn;
class Rknpu3IoBinding;
class Rknpu3Session;
} // namespace ztu::rk

namespace ztu::somemodelruntime::rknn {

class Session;
class IoBinding;

class ZTU_SOMEMODELRUNTIME_RKNN_API Exception : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

enum class TensorElementType {
  Unknown,
  Float32,
  Float16,
  Int8,
  Uint8,
  Int16,
  Uint16,
  Int32,
  Uint32,
  Int64,
  Bool,
};

struct Float16_t {
  uint16_t value = 0;
};

enum class DeviceType { Cpu, Rknpu2, Rknpu3 };
enum class ExecutionProvider { Auto, Rknpu2, Rknpu3 };
enum class IoKind { Input, Output };
enum class Layout { Nchw, NchwSoftware, Nhwc, Any };
enum class ZeroCopyLayout { Nhwc, Native };
enum class CoreMask { Auto, All, Core0, Core1, Core2, Core01, Core012 };

struct ZTU_SOMEMODELRUNTIME_RKNN_API RknnProviderOptions {
  Layout layout = Layout::Nchw;
  ZeroCopyLayout zero_copy_layout = ZeroCopyLayout::Nhwc;
  size_t max_queue_size = 3;
  int threads_per_core = 1;
  int64_t submit_timeout_ms = 10000;
  bool sequential_callbacks = true;
  std::optional<std::vector<uint64_t>> schedule;
  std::optional<CoreMask> tp_mode;
  bool enable_pacing = false;
  bool disable_dup_context = false;
  std::vector<std::string> custom_op_paths;
  bool custom_op_default_path = false;
};

using Rknpu2ProviderOptions = RknnProviderOptions;

struct ZTU_SOMEMODELRUNTIME_RKNN_API Rknpu3ProviderOptions {
  std::optional<std::string> runtime_library;
  std::optional<std::string> device_id;
  std::optional<std::string> weight_path;
  int32_t priority = 0;
  uint32_t run_timeout_ms = 0;
  uint32_t run_core_mask = 0;
};

class ZTU_SOMEMODELRUNTIME_RKNN_API Env {
public:
  struct Impl;
  using WarningHandler = std::function<void(std::string_view)>;

  explicit Env(std::string log_id = "ztu_somemodelruntime_rknn");
  void SetWarningHandler(WarningHandler handler);

private:
  std::shared_ptr<Impl> impl_;
  friend class Session;
};

class ZTU_SOMEMODELRUNTIME_RKNN_API SessionOptions {
public:
  SessionOptions();
  SessionOptions &SetExecutionProvider(ExecutionProvider provider);
  ExecutionProvider GetExecutionProvider() const;
  SessionOptions &SetRknnProviderOptions(const RknnProviderOptions &options);
  const RknnProviderOptions &GetRknnProviderOptions() const;
  SessionOptions &
  SetRknpu3ProviderOptions(const Rknpu3ProviderOptions &options);
  const Rknpu3ProviderOptions &GetRknpu3ProviderOptions() const;

private:
  ExecutionProvider provider_ = ExecutionProvider::Auto;
  RknnProviderOptions options_;
  Rknpu3ProviderOptions rknpu3_options_;
};

class ZTU_SOMEMODELRUNTIME_RKNN_API RunOptions {
public:
  RunOptions() = default;
  RunOptions &AddConfigEntry(std::string key, std::string value);
  RunOptions &SetDispatchBatch(bool enabled);
  bool GetDispatchBatch() const;

private:
  std::map<std::string, std::string> entries_;
};

class ZTU_SOMEMODELRUNTIME_RKNN_API MemoryInfo {
public:
  static MemoryInfo CreateCpu(int device_id = 0);
  static MemoryInfo CreateRknpu2(int device_id = 0);
  static MemoryInfo CreateRknpu3(int device_id = 0);

  DeviceType GetDeviceType() const noexcept { return device_type_; }
  int GetDeviceId() const noexcept { return device_id_; }
  const char *GetName() const noexcept;

private:
  MemoryInfo(DeviceType type, int device_id)
      : device_type_(type), device_id_(device_id) {}
  DeviceType device_type_;
  int device_id_;
};

class ZTU_SOMEMODELRUNTIME_RKNN_API TensorTypeAndShapeInfo {
public:
  TensorTypeAndShapeInfo() = default;
  TensorTypeAndShapeInfo(TensorElementType type, std::vector<int64_t> shape);
  TensorElementType GetElementType() const noexcept { return type_; }
  const std::vector<int64_t> &GetShape() const noexcept { return shape_; }
  size_t GetDimensionsCount() const noexcept { return shape_.size(); }
  size_t GetElementCount() const;

private:
  TensorElementType type_ = TensorElementType::Unknown;
  std::vector<int64_t> shape_;
};

class ZTU_SOMEMODELRUNTIME_RKNN_API TypeInfo {
public:
  TypeInfo() = default;
  explicit TypeInfo(TensorTypeAndShapeInfo tensor_info);
  const TensorTypeAndShapeInfo &GetTensorTypeAndShapeInfo() const noexcept;

private:
  TensorTypeAndShapeInfo tensor_info_;
};

struct ZTU_SOMEMODELRUNTIME_RKNN_API NodeArg {
  std::string name;
  std::vector<int64_t> shape;
  std::string type;
  std::string layout;
  std::vector<uint64_t> strides;
};

struct ZTU_SOMEMODELRUNTIME_RKNN_API ModelMetadata {
  std::map<std::string, std::string> custom_metadata_map;
};

struct ZTU_SOMEMODELRUNTIME_RKNN_API ValueMemoryInfo {
  DeviceType device_type = DeviceType::Cpu;
  int device_id = 0;
  size_t size = 0;
  int fd = -1;
  int32_t offset = 0;
  uintptr_t virt_addr = 0;
  size_t import_count = 0;
  bool dmabuf_backed = false;
  uint32_t attr_size = 0;
  uint32_t size_with_stride = 0;
  uint32_t w_stride = 0;
  uint32_t h_stride = 0;
  int fmt = 0;
  bool pass_through = false;
  uint64_t aligned_size = 0;
  int32_t core_id = -1;
  std::string layout;
  std::vector<uint64_t> strides;
};

class ZTU_SOMEMODELRUNTIME_RKNN_API AllocatorWithDefaultOptions {
public:
  AllocatorWithDefaultOptions() = default;
};

class ZTU_SOMEMODELRUNTIME_RKNN_API Value {
public:
  Value() = default;
  Value(const Value &) = delete;
  Value &operator=(const Value &) = delete;
  Value(Value &&) noexcept = default;
  Value &operator=(Value &&) noexcept = default;
  ~Value() = default;

  template <typename T>
  static Value CreateTensor(const MemoryInfo &memory_info, T *data,
                            size_t element_count, const int64_t *shape,
                            size_t shape_len) {
    return CreateTensorImpl(memory_info, data, element_count * sizeof(T), shape,
                            shape_len, ElementTypeFor<T>());
  }

  static Value CreateTensor(const MemoryInfo &memory_info, void *data,
                            size_t bytes, const int64_t *shape,
                            size_t shape_len, TensorElementType type);

  template <typename T>
  static Value CreateTensor(AllocatorWithDefaultOptions &allocator,
                            const int64_t *shape, size_t shape_len) {
    (void)allocator;
    return CreateOwnedTensor(shape, shape_len, ElementTypeFor<T>());
  }

  static Value CreateOwnedTensor(const int64_t *shape, size_t shape_len,
                                 TensorElementType type);

  bool IsTensor() const noexcept;
  bool HasValue() const noexcept;
  void *GetTensorMutableRawData();
  const void *GetTensorRawData() const;
  template <typename T> T *GetTensorMutableData() {
    return static_cast<T *>(GetTensorMutableRawData());
  }
  template <typename T> const T *GetTensorData() const {
    return static_cast<const T *>(GetTensorRawData());
  }
  size_t GetTensorSizeInBytes() const;
  TensorTypeAndShapeInfo GetTensorTypeAndShapeInfo() const;
  ValueMemoryInfo GetMemoryInfo() const;
  void Update(const void *data, size_t bytes,
              const std::vector<int64_t> &shape, TensorElementType type);
  void CopyTo(void *data, size_t bytes) const;
  void SyncToDevice();
  void SyncFromDevice();

private:
  explicit Value(std::shared_ptr<ztu::rk::RknnOrtValue> impl)
      : impl_(std::move(impl)) {}
  static Value CreateTensorImpl(const MemoryInfo &memory_info, void *data,
                                size_t bytes, const int64_t *shape,
                                size_t shape_len, TensorElementType type);

  template <typename T> static constexpr TensorElementType ElementTypeFor() {
    using U = std::remove_cv_t<T>;
    if constexpr (std::is_same_v<U, float>)
      return TensorElementType::Float32;
    else if constexpr (std::is_same_v<U, Float16_t>)
      return TensorElementType::Float16;
    else if constexpr (std::is_same_v<U, int8_t>)
      return TensorElementType::Int8;
    else if constexpr (std::is_same_v<U, uint8_t>)
      return TensorElementType::Uint8;
    else if constexpr (std::is_same_v<U, int16_t>)
      return TensorElementType::Int16;
    else if constexpr (std::is_same_v<U, uint16_t>)
      return TensorElementType::Uint16;
    else if constexpr (std::is_same_v<U, int32_t>)
      return TensorElementType::Int32;
    else if constexpr (std::is_same_v<U, uint32_t>)
      return TensorElementType::Uint32;
    else if constexpr (std::is_same_v<U, int64_t>)
      return TensorElementType::Int64;
    else if constexpr (std::is_same_v<U, bool>)
      return TensorElementType::Bool;
    else {
      static_assert(!sizeof(T), "unsupported tensor element type");
    }
  }

  std::shared_ptr<ztu::rk::RknnOrtValue> impl_;
  friend struct ValueAccess;
  friend class Session;
  friend class IoBinding;
};

struct ValueAccess;

using RunAsyncCallbackFn =
    void (*)(void *user_data, Value *outputs, size_t num_outputs,
             const char *error_message);

class ZTU_SOMEMODELRUNTIME_RKNN_API Session {
public:
  struct Impl;
  Session(const Env &env, const char *model_path,
          const SessionOptions &options = SessionOptions(),
          const Session *use_weight_from = nullptr);
  Session(const Env &env, const void *model_data, size_t model_size,
          const SessionOptions &options = SessionOptions(),
          const Session *use_weight_from = nullptr);
  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;
  Session(Session &&) noexcept = default;
  Session &operator=(Session &&other) noexcept;
  ~Session();

  std::vector<Value>
  Run(const RunOptions &run_options, const char *const *input_names,
      const Value *input_values, size_t input_count,
      const char *const *output_names, size_t output_count);
  void Run(const RunOptions &run_options, const char *const *input_names,
           const Value *input_values, size_t input_count,
           const char *const *output_names, Value *output_values,
           size_t output_count);
  void RunAsync(const RunOptions &run_options,
                const char *const *input_names, const Value *input_values,
                size_t input_count, const char *const *output_names,
                Value *output_values, size_t output_count,
                RunAsyncCallbackFn callback, void *user_data);
  std::optional<std::vector<Value>>
  RunPipeline(const char *const *input_names, const Value *input_values,
              size_t input_count, size_t depth = 3, bool reset = false);
  void Run(const RunOptions &run_options, IoBinding &binding);

  size_t GetInputCount() const;
  size_t GetOutputCount() const;
  const std::string &GetInputName(size_t index) const;
  const std::string &GetOutputName(size_t index) const;
  TypeInfo GetInputTypeInfo(size_t index) const;
  TypeInfo GetOutputTypeInfo(size_t index) const;
  std::vector<NodeArg> GetInputs() const;
  std::vector<NodeArg> GetOutputs() const;
  std::vector<NodeArg> GetNativeInputs() const;
  std::vector<NodeArg> GetNativeOutputs() const;
  ModelMetadata GetModelMetadata() const;
  const char *GetExecutionProviderName() const noexcept;
  DeviceType GetDeviceType() const noexcept;

  Value CreateIoValue(std::string_view name, IoKind kind,
                      const MemoryInfo &memory_info,
                      TensorElementType type = TensorElementType::Unknown,
                      std::optional<std::vector<int64_t>> shape = std::nullopt,
                      uint64_t memory_flags = 0);
  Value CreateIoValueFromDmaBuf(
      std::string_view name, IoKind kind, int fd, size_t size,
      TensorElementType type, std::optional<std::vector<int64_t>> shape,
      int32_t offset = 0, void *virt_addr = nullptr);

private:
  std::shared_ptr<Impl> impl_;
  friend class IoBinding;
};

class ZTU_SOMEMODELRUNTIME_RKNN_API IoBinding {
public:
  explicit IoBinding(Session &session);
  IoBinding(const IoBinding &) = delete;
  IoBinding &operator=(const IoBinding &) = delete;
  IoBinding(IoBinding &&) noexcept = default;
  IoBinding &operator=(IoBinding &&) noexcept = default;
  ~IoBinding();

  void BindInput(const char *name, const Value &value);
  void BindOutput(const char *name, const Value &value);
  void BindOutput(const char *name, const MemoryInfo &memory_info);
  void ClearBoundInputs();
  void ClearBoundOutputs();
  void SynchronizeInputs();
  void SynchronizeOutputs();
  std::vector<Value> GetOutputValues() const;

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
  friend class Session;
};

ZTU_SOMEMODELRUNTIME_RKNN_API const char *TensorElementTypeName(TensorElementType type) noexcept;

} // namespace ztu::somemodelruntime::rknn
