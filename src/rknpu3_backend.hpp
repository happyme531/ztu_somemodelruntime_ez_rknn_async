#pragma once

#include "ez_rknn_zero_copy.hpp"
#include "rknpu3_abi.hpp"

#include <ztu/somemodelruntime_rknn.hpp>

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ztu::rk {

class Rknpu3IoBinding;

class Rknpu3Session : public std::enable_shared_from_this<Rknpu3Session> {
public:
  using WarningHandler = std::function<void(const std::string &)>;

  Rknpu3Session(
      const std::string &model_path,
      const somemodelruntime::rknn::Rknpu3ProviderOptions &options,
      WarningHandler warning_handler = {});
  Rknpu3Session(
      std::vector<uint8_t> model_data,
      const somemodelruntime::rknn::Rknpu3ProviderOptions &options,
      WarningHandler warning_handler = {});
  ~Rknpu3Session();

  Rknpu3Session(const Rknpu3Session &) = delete;
  Rknpu3Session &operator=(const Rknpu3Session &) = delete;

  const std::vector<rknpu3_abi::TensorAttr> &input_attrs() const;
  const std::vector<rknpu3_abi::TensorAttr> &output_attrs() const;
  const std::vector<std::string> &input_names() const;
  const std::vector<std::string> &output_names() const;
  const std::string &custom_string() const;
  const std::string &runtime_version() const;

  size_t input_index(const std::string &name) const;
  size_t output_index(const std::string &name) const;

  std::shared_ptr<RknnOrtValue> create_value_for_io(
      const std::string &name, bool is_input, rknn_tensor_type requested_type,
      uint64_t alloc_flags,
      const std::optional<std::vector<int64_t>> &requested_shape =
          std::nullopt);
  std::shared_ptr<RknnOrtValue> create_value_from_fd(
      const std::string &name, bool is_input, rknn_tensor_type requested_type,
      int fd, void *virt_addr, uint64_t size, int32_t offset,
      const std::optional<std::vector<int64_t>> &requested_shape =
          std::nullopt);

  void run_with_iobinding(Rknpu3IoBinding &binding);

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
  friend class Rknpu3IoBinding;
};

class Rknpu3IoBinding {
public:
  explicit Rknpu3IoBinding(std::shared_ptr<Rknpu3Session> session);

  void bind_cpu_input(const std::string &name,
                      std::shared_ptr<RknnOrtValue> cpu_value);
  void bind_ortvalue_input(const std::string &name,
                           std::shared_ptr<RknnOrtValue> value);
  void bind_ortvalue_output(const std::string &name,
                            std::shared_ptr<RknnOrtValue> value);
  void bind_output(const std::string &name, const std::string &device_type,
                   int device_id, std::optional<rknn_tensor_type> element_type,
                   const std::optional<std::vector<int64_t>> &shape,
                   std::optional<uint64_t> buffer_ptr, uint64_t alloc_flags);

  std::vector<std::shared_ptr<RknnOrtValue>> get_outputs() const;
  void clear_binding_inputs();
  void clear_binding_outputs();
  void synchronize_inputs();
  void synchronize_outputs();

private:
  friend class Rknpu3Session;
  void remember_output_order(size_t index);

  std::shared_ptr<Rknpu3Session> session_;
  std::map<size_t, std::shared_ptr<RknnOrtValue>> bound_inputs_;
  std::map<size_t, std::shared_ptr<RknnOrtValue>> bound_outputs_;
  std::map<size_t, bool> cpu_output_indices_;
  std::vector<size_t> output_order_;
};

std::vector<int64_t>
rknpu3_attr_shape(const rknpu3_abi::TensorAttr &attr);
rknn_tensor_type
rknpu3_attr_rknpu2_compatible_type(const rknpu3_abi::TensorAttr &attr);
std::string rknpu3_layout_name(rknpu3_abi::TensorLayout layout);
std::vector<uint64_t>
rknpu3_attr_strides(const rknpu3_abi::TensorAttr &attr);

} // namespace ztu::rk
