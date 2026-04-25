#pragma once

#include "rknn_api.h"
#include "rknn_error_utils.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cctype>
#include <cstdlib>
#include <fcntl.h>
#include <iostream>
#if defined(__has_include)
#if __has_include(<linux/dma-heap.h>)
#include <linux/dma-heap.h>
#define ZTU_HAVE_LINUX_DMA_HEAP_H 1
#endif
#endif
#ifndef ZTU_HAVE_LINUX_DMA_HEAP_H
#include <linux/ioctl.h>
#include <linux/types.h>
struct dma_heap_allocation_data {
  __u64 len;
  __u32 fd;
  __u32 fd_flags;
  __u64 heap_flags;
};
#define DMA_HEAP_IOC_MAGIC 'H'
#define DMA_HEAP_IOCTL_ALLOC                                                   \
  _IOWR(DMA_HEAP_IOC_MAGIC, 0x0, struct dma_heap_allocation_data)
#endif
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace ztu {
namespace rk {
namespace py = pybind11;

class ZeroCopyEzRknn;
class RknnIoBinding;

inline size_t rknn_tensor_type_size(rknn_tensor_type type) {
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
    throw std::runtime_error("Unsupported RKNN tensor type");
  }
}

inline py::dtype rknn_tensor_type_to_numpy_dtype(rknn_tensor_type type) {
  switch (type) {
  case RKNN_TENSOR_FLOAT32:
    return py::dtype::of<float>();
  case RKNN_TENSOR_FLOAT16:
    return py::dtype("float16");
  case RKNN_TENSOR_INT8:
    return py::dtype::of<int8_t>();
  case RKNN_TENSOR_UINT8:
    return py::dtype::of<uint8_t>();
  case RKNN_TENSOR_INT16:
    return py::dtype::of<int16_t>();
  case RKNN_TENSOR_UINT16:
    return py::dtype::of<uint16_t>();
  case RKNN_TENSOR_INT32:
    return py::dtype::of<int32_t>();
  case RKNN_TENSOR_UINT32:
    return py::dtype::of<uint32_t>();
  case RKNN_TENSOR_INT64:
    return py::dtype::of<int64_t>();
  case RKNN_TENSOR_BOOL:
    return py::dtype::of<bool>();
  default:
    throw std::runtime_error("Unsupported RKNN tensor type");
  }
}

inline std::vector<int64_t> attr_shape_i64(const rknn_tensor_attr &attr) {
  std::vector<int64_t> shape;
  shape.reserve(attr.n_dims);
  for (uint32_t i = 0; i < attr.n_dims; ++i) {
    shape.push_back(static_cast<int64_t>(attr.dims[i]));
  }
  return shape;
}

inline std::string shape_to_string(const std::vector<int64_t> &shape) {
  std::string text = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      text += ", ";
    }
    text += std::to_string(shape[i]);
  }
  text += "]";
  return text;
}

inline void require_shape_matches(const std::vector<int64_t> &actual,
                                  const std::vector<int64_t> &expected,
                                  const std::string &context) {
  if (actual != expected) {
    throw std::runtime_error(context + " shape mismatch: expected " +
                             shape_to_string(expected) + ", got " +
                             shape_to_string(actual));
  }
}

inline std::vector<int64_t> numpy_shape_i64(const py::array &array) {
  std::vector<int64_t> shape;
  shape.reserve(array.ndim());
  for (ssize_t i = 0; i < array.ndim(); ++i) {
    shape.push_back(static_cast<int64_t>(array.shape(i)));
  }
  return shape;
}

inline std::string rknn_tensor_type_name(rknn_tensor_type type) {
  switch (type) {
  case RKNN_TENSOR_FLOAT32:
    return "float32";
  case RKNN_TENSOR_FLOAT16:
    return "float16";
  case RKNN_TENSOR_INT8:
    return "int8";
  case RKNN_TENSOR_UINT8:
    return "uint8";
  case RKNN_TENSOR_INT16:
    return "int16";
  case RKNN_TENSOR_UINT16:
    return "uint16";
  case RKNN_TENSOR_INT32:
    return "int32";
  case RKNN_TENSOR_UINT32:
    return "uint32";
  case RKNN_TENSOR_INT64:
    return "int64";
  case RKNN_TENSOR_BOOL:
    return "bool";
  default:
    return "unknown(" + std::to_string(static_cast<int>(type)) + ")";
  }
}

inline void require_type_matches(rknn_tensor_type actual,
                                 rknn_tensor_type expected,
                                 const std::string &context) {
  if (actual != expected) {
    throw std::runtime_error(context + " element type mismatch: expected " +
                             rknn_tensor_type_name(expected) + ", got " +
                             rknn_tensor_type_name(actual));
  }
}

inline void append_attr_mismatch(std::vector<std::string> &items,
                                 const std::string &name,
                                 uint64_t actual, uint64_t expected) {
  if (actual != expected) {
    items.push_back(name + " expected " + std::to_string(expected) +
                    ", got " + std::to_string(actual));
  }
}

inline void append_attr_mismatch_signed(std::vector<std::string> &items,
                                        const std::string &name,
                                        int64_t actual, int64_t expected) {
  if (actual != expected) {
    items.push_back(name + " expected " + std::to_string(expected) +
                    ", got " + std::to_string(actual));
  }
}

inline uint32_t attr_width_stride_value(const rknn_tensor_attr &attr) {
  if (attr.w_stride > 0) {
    return attr.w_stride;
  }
  if (attr.n_dims == 4 && attr.fmt == RKNN_TENSOR_NHWC) {
    return attr.dims[2];
  }
  if (attr.n_dims == 4 && attr.fmt == RKNN_TENSOR_NCHW) {
    return attr.dims[3];
  }
  return 0;
}

inline uint32_t attr_height_stride_value(const rknn_tensor_attr &attr) {
  if (attr.h_stride > 0) {
    return attr.h_stride;
  }
  if (attr.n_dims == 4 && attr.fmt == RKNN_TENSOR_NHWC) {
    return attr.dims[1];
  }
  if (attr.n_dims == 4 && attr.fmt == RKNN_TENSOR_NCHW) {
    return attr.dims[2];
  }
  return 0;
}

inline std::string native_attr_mismatch_reason(const rknn_tensor_attr &actual,
                                               const rknn_tensor_attr &expected) {
  std::vector<std::string> items;
  append_attr_mismatch(items, "type", static_cast<uint64_t>(actual.type),
                       static_cast<uint64_t>(expected.type));
  append_attr_mismatch(items, "n_dims", actual.n_dims, expected.n_dims);
  const uint32_t dims_to_compare = std::max(actual.n_dims, expected.n_dims);
  for (uint32_t i = 0; i < dims_to_compare && i < RKNN_MAX_DIMS; ++i) {
    append_attr_mismatch(items, "dims[" + std::to_string(i) + "]",
                         actual.dims[i], expected.dims[i]);
  }
  append_attr_mismatch(items, "fmt", static_cast<uint64_t>(actual.fmt),
                       static_cast<uint64_t>(expected.fmt));
  append_attr_mismatch(items, "n_elems", actual.n_elems, expected.n_elems);
  append_attr_mismatch(items, "size", actual.size, expected.size);
  append_attr_mismatch(items, "size_with_stride", actual.size_with_stride,
                       expected.size_with_stride);
  append_attr_mismatch(items, "w_stride", attr_width_stride_value(actual),
                       attr_width_stride_value(expected));
  append_attr_mismatch(items, "h_stride", attr_height_stride_value(actual),
                       attr_height_stride_value(expected));
  append_attr_mismatch(items, "pass_through", actual.pass_through,
                       expected.pass_through);
  append_attr_mismatch(items, "qnt_type", static_cast<uint64_t>(actual.qnt_type),
                       static_cast<uint64_t>(expected.qnt_type));
  append_attr_mismatch_signed(items, "fl", static_cast<int64_t>(actual.fl),
                              static_cast<int64_t>(expected.fl));
  append_attr_mismatch_signed(items, "zp", static_cast<int64_t>(actual.zp),
                              static_cast<int64_t>(expected.zp));
  if (actual.scale != expected.scale) {
    items.push_back("scale expected " + std::to_string(expected.scale) +
                    ", got " + std::to_string(actual.scale));
  }
  if (items.empty()) {
    return "";
  }
  std::string reason;
  for (size_t i = 0; i < items.size(); ++i) {
    if (i > 0) {
      reason += "; ";
    }
    reason += items[i];
  }
  return reason;
}

inline void require_native_attr_matches(const rknn_tensor_attr &actual,
                                        const rknn_tensor_attr &expected,
                                        const std::string &context) {
  const std::string reason = native_attr_mismatch_reason(actual, expected);
  if (!reason.empty()) {
    throw std::runtime_error(
        context +
        " native zero-copy layout mismatch: " + reason +
        ". The OrtValue buffer was written using a different RKNN native "
        "layout than this model input expects. Create the OrtValue with this "
        "consumer session/name/io_kind, or use a copy/transpose path instead "
        "of binding the dmabuf directly.");
  }
}

inline uint32_t attr_dense_bytes(const rknn_tensor_attr &attr) {
  return static_cast<uint32_t>(attr.n_elems * rknn_tensor_type_size(attr.type));
}

inline uint32_t attr_alloc_bytes(const rknn_tensor_attr &attr) {
  if (attr.n_dims != 4 || attr.fmt == RKNN_TENSOR_UNDEFINED) {
    return attr_dense_bytes(attr); //FIXME: zt: rknpu2 api returned wrong size for non-4d tensor?
  }
  if (attr.size_with_stride > 0) {
    return attr.size_with_stride;
  }
  if (attr.size > 0) {
    return attr.size;
  }
  return attr_dense_bytes(attr);
}

inline uint32_t attr_width_stride(const rknn_tensor_attr &attr) {
  if (attr.w_stride > 0) {
    return attr.w_stride;
  }
  if (attr.n_dims == 4 && attr.fmt == RKNN_TENSOR_NHWC) {
    return attr.dims[2];
  }
  if (attr.n_dims == 4 && attr.fmt == RKNN_TENSOR_NCHW) {
    return attr.dims[3];
  }
  return 0;
}

inline uint32_t attr_height_stride(const rknn_tensor_attr &attr) {
  if (attr.h_stride > 0) {
    return attr.h_stride;
  }
  if (attr.n_dims == 4 && attr.fmt == RKNN_TENSOR_NHWC) {
    return attr.dims[1];
  }
  if (attr.n_dims == 4 && attr.fmt == RKNN_TENSOR_NCHW) {
    return attr.dims[2];
  }
  return 0;
}

inline bool supports_native_nhwc_query_type(rknn_tensor_type type) {
  return type == RKNN_TENSOR_FLOAT32 || type == RKNN_TENSOR_FLOAT16 ||
         type == RKNN_TENSOR_INT8 || type == RKNN_TENSOR_UINT8;
}

enum class ZeroCopyInputLayout { NCHW, NHWC, ANY };

inline rknn_tensor_attr
make_user_input_attr_for_layout(const rknn_tensor_attr &attr,
                                ZeroCopyInputLayout layout) {
  if (attr.n_dims != 4) {
    return attr;
  }
  rknn_tensor_attr logical = attr;
  const uint32_t n = attr.dims[0];
  if (layout == ZeroCopyInputLayout::NHWC) {
    return logical;
  }
  if (attr.fmt == RKNN_TENSOR_NHWC) {
    logical.dims[0] = n;
    logical.dims[1] = attr.dims[3];
    logical.dims[2] = attr.dims[1];
    logical.dims[3] = attr.dims[2];
    logical.fmt = RKNN_TENSOR_NCHW;
  }
  return logical;
}

inline bool can_bind_nc1hwc2_input_as_nhwc(const rknn_tensor_attr &native_attr,
                                           const rknn_tensor_attr &logical_attr) {
  return logical_attr.n_dims == 4 && native_attr.fmt == RKNN_TENSOR_NC1HWC2 &&
         native_attr.pass_through == 0 &&
         supports_native_nhwc_query_type(native_attr.type);
}

inline rknn_tensor_attr make_nhwc_input_attr_from_logical(
    const rknn_tensor_attr &native_attr, const rknn_tensor_attr &logical_attr) {
  rknn_tensor_attr attr = native_attr;
  const uint32_t n = logical_attr.dims[0];
  uint32_t c = logical_attr.dims[1];
  uint32_t h = logical_attr.dims[2];
  uint32_t w = logical_attr.dims[3];
  if (logical_attr.fmt == RKNN_TENSOR_NHWC) {
    h = logical_attr.dims[1];
    w = logical_attr.dims[2];
    c = logical_attr.dims[3];
  }
  const uint32_t w_stride = native_attr.w_stride > 0 ? native_attr.w_stride : w;
  const uint32_t h_stride = native_attr.h_stride > 0 ? native_attr.h_stride : h;
  attr.n_dims = 4;
  std::memset(attr.dims, 0, sizeof(attr.dims));
  attr.dims[0] = n;
  attr.dims[1] = h;
  attr.dims[2] = w;
  attr.dims[3] = c;
  attr.n_elems = n * h * w * c;
  attr.fmt = RKNN_TENSOR_NHWC;
  attr.type = native_attr.type;
  attr.size =
      static_cast<uint32_t>(attr.n_elems * rknn_tensor_type_size(attr.type));
  attr.w_stride = w_stride;
  attr.h_stride = native_attr.h_stride;
  attr.size_with_stride = static_cast<uint32_t>(
      static_cast<uint64_t>(n) * h_stride * w_stride * c *
      rknn_tensor_type_size(attr.type));
  attr.pass_through = 0;
  return attr;
}

inline void require_dense_native_copy_supported(const rknn_tensor_attr &attr,
                                                const char *op_name) {
  const uint32_t dense = attr_dense_bytes(attr);
  const uint32_t alloc = attr_alloc_bytes(attr);
  if (dense != alloc) {
    throw std::runtime_error(std::string(op_name) +
                             " for strided RKNN native tensors is not "
                             "implemented yet; use a native OrtValue without "
                             "copying to CPU, or bind a model/output whose "
                             "size_with_stride equals dense size");
  }
}

struct RknnContextHolder {
  explicit RknnContextHolder(rknn_context context, bool owns_context = true,
                             std::shared_ptr<void> owner = nullptr)
      : ctx(context), owns_context(owns_context), owner(std::move(owner)) {}
  RknnContextHolder(const RknnContextHolder &) = delete;
  RknnContextHolder &operator=(const RknnContextHolder &) = delete;
  ~RknnContextHolder() {
    if (owns_context && ctx != 0) {
      rknn_destroy(ctx);
      ctx = 0;
    }
  }

  rknn_context ctx = 0;
  bool owns_context = true;
  std::shared_ptr<void> owner;
};

class SharedDmaBuffer {
public:
  SharedDmaBuffer() = default;
  SharedDmaBuffer(const SharedDmaBuffer &) = delete;
  SharedDmaBuffer &operator=(const SharedDmaBuffer &) = delete;
  ~SharedDmaBuffer() {
    if (owns_mapping_ && virt_addr_ != nullptr && map_size_ > 0) {
      munmap(virt_addr_, map_size_);
    }
    if (owns_fd_ && fd_ >= 0) {
      close(fd_);
    }
  }

  static std::shared_ptr<SharedDmaBuffer> allocate(size_t size) {
    static const char *kHeapPaths[] = {
        "/dev/dma_heap/system"};
    int heap_fd = -1;
    for (const char *path : kHeapPaths) {
      heap_fd = open(path, O_RDONLY | O_CLOEXEC);
      if (heap_fd >= 0) {
        break;
      }
    }
    if (heap_fd < 0) {
      return nullptr;
    }

    dma_heap_allocation_data data = {};
    data.len = size;
    data.fd_flags = O_RDWR | O_CLOEXEC;
    data.heap_flags = 0;
    if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &data) < 0) {
      close(heap_fd);
      return nullptr;
    }
    close(heap_fd);

    void *virt =
        mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, data.fd, 0);
    if (virt == MAP_FAILED) {
      close(static_cast<int>(data.fd));
      return nullptr;
    }

    auto buffer = std::shared_ptr<SharedDmaBuffer>(new SharedDmaBuffer());
    buffer->fd_ = static_cast<int>(data.fd);
    buffer->virt_addr_ = virt;
    buffer->size_ = size;
    buffer->map_size_ = size;
    buffer->offset_ = 0;
    buffer->owns_fd_ = true;
    buffer->owns_mapping_ = true;
    return buffer;
  }

  static std::shared_ptr<SharedDmaBuffer> from_fd(int fd, void *virt_addr,
                                                  size_t size,
                                                  int32_t offset) {
    int owned_fd = dup(fd);
    if (owned_fd < 0) {
      throw std::runtime_error("dup dma-buf fd failed");
    }
    void *mapped = virt_addr;
    bool owns_mapping = false;
    size_t map_size = size + static_cast<size_t>(std::max<int32_t>(offset, 0));
    if (mapped == nullptr) {
      mapped =
          mmap(nullptr, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, owned_fd,
               0);
      if (mapped == MAP_FAILED) {
        close(owned_fd);
        throw std::runtime_error("mmap dma-buf fd failed");
      }
      owns_mapping = true;
    }

    auto buffer = std::shared_ptr<SharedDmaBuffer>(new SharedDmaBuffer());
    buffer->fd_ = owned_fd;
    buffer->virt_addr_ = mapped;
    buffer->size_ = size;
    buffer->map_size_ = map_size;
    buffer->offset_ = offset;
    buffer->owns_fd_ = true;
    buffer->owns_mapping_ = owns_mapping;
    return buffer;
  }

  int fd() const { return fd_; }
  void *virt_addr() const { return virt_addr_; }
  size_t size() const { return size_; }
  int32_t offset() const { return offset_; }

private:
  int fd_ = -1;
  void *virt_addr_ = nullptr;
  size_t size_ = 0;
  size_t map_size_ = 0;
  int32_t offset_ = 0;
  bool owns_fd_ = false;
  bool owns_mapping_ = false;
};

class RknnOrtValue {
public:
  RknnOrtValue() = default;
  RknnOrtValue(const RknnOrtValue &) = delete;
  RknnOrtValue &operator=(const RknnOrtValue &) = delete;
  RknnOrtValue(RknnOrtValue &&) = default;
  RknnOrtValue &operator=(RknnOrtValue &&) = default;

  ~RknnOrtValue() {
    for (auto &item : imports_) {
      if (item.second.mem != nullptr && item.second.holder &&
          item.second.holder->ctx != 0) {
        rknn_destroy_mem(item.second.holder->ctx, item.second.mem);
        item.second.mem = nullptr;
      }
    }
  }

  static std::shared_ptr<RknnOrtValue> from_cpu_array(py::array array) {
    auto value = std::shared_ptr<RknnOrtValue>(new RknnOrtValue());
    value->device_type_ = "cpu";
    value->cpu_array_ = py::array::ensure(array, py::array::c_style);
    if (!value->cpu_array_) {
      throw std::runtime_error("Input must be a contiguous numpy array");
    }
    value->shape_.reserve(value->cpu_array_.ndim());
    for (ssize_t i = 0; i < value->cpu_array_.ndim(); ++i) {
      value->shape_.push_back(static_cast<int64_t>(value->cpu_array_.shape(i)));
    }
    value->type_ = RKNN_TENSOR_TYPE_MAX;
    return value;
  }

  static std::shared_ptr<RknnOrtValue>
  create_rknn(std::shared_ptr<RknnContextHolder> holder,
              const rknn_tensor_attr &attr, std::vector<int64_t> logical_shape,
              rknn_tensor_type logical_type, uint64_t alloc_flags) {
    if (!holder || holder->ctx == 0) {
      throw std::runtime_error("RKNN context is not available");
    }
    (void)alloc_flags;
    const uint32_t bytes = attr_alloc_bytes(attr);
    auto value = std::shared_ptr<RknnOrtValue>(new RknnOrtValue());
    value->device_type_ = "rknpu2";
    value->buffer_ = SharedDmaBuffer::allocate(bytes);
    value->attr_ = attr;
    value->shape_ = std::move(logical_shape);
    value->type_ = logical_type;
    if (value->buffer_) {
      value->import_for(holder);
    } else {
      value->create_fallback_mem(holder, bytes);
    }
    return value;
  }

  static std::shared_ptr<RknnOrtValue>
  create_generic_rknn(std::vector<int64_t> shape, rknn_tensor_type type,
                      uint64_t alloc_flags) {
    (void)alloc_flags;
    rknn_tensor_attr attr = {};
    attr.n_dims = static_cast<uint32_t>(shape.size());
    attr.type = type;
    attr.fmt = shape.size() == 4 ? RKNN_TENSOR_NCHW : RKNN_TENSOR_UNDEFINED;
    attr.n_elems = 1;
    for (size_t i = 0; i < shape.size() && i < RKNN_MAX_DIMS; ++i) {
      attr.dims[i] = static_cast<uint32_t>(shape[i]);
      attr.n_elems *= static_cast<uint32_t>(shape[i]);
    }
    attr.size = static_cast<uint32_t>(attr.n_elems *
                                      rknn_tensor_type_size(attr.type));
    attr.size_with_stride = attr.size;
    auto value = std::shared_ptr<RknnOrtValue>(new RknnOrtValue());
    value->device_type_ = "rknpu2";
    value->buffer_ = SharedDmaBuffer::allocate(attr.size_with_stride);
    if (!value->buffer_) {
      throw std::runtime_error(
          "Failed to allocate dma-buf for generic RKNN OrtValue");
    }
    value->attr_ = attr;
    value->shape_ = std::move(shape);
    value->type_ = type;
    return value;
  }

  static std::shared_ptr<RknnOrtValue>
  from_fd(std::shared_ptr<RknnContextHolder> holder, int32_t fd,
          void *virt_addr, uint32_t size, int32_t offset,
          const rknn_tensor_attr &attr, std::vector<int64_t> logical_shape,
          rknn_tensor_type logical_type) {
    if (!holder || holder->ctx == 0) {
      throw std::runtime_error("RKNN context is not available");
    }
    auto value = std::shared_ptr<RknnOrtValue>(new RknnOrtValue());
    value->device_type_ = "rknpu2";
    value->buffer_ =
        SharedDmaBuffer::from_fd(fd, virt_addr, static_cast<size_t>(size),
                                 offset);
    value->attr_ = attr;
    value->shape_ = std::move(logical_shape);
    value->type_ = logical_type;
    value->import_for(holder);
    return value;
  }

  bool is_rknn() const { return device_type_ == "rknpu2"; }
  bool is_cpu() const { return device_type_ == "cpu"; }
  bool has_value() const { return is_cpu() || buffer_ || !imports_.empty(); }
  bool is_tensor() const { return has_value(); }
  const std::string &device_type() const { return device_type_; }
  const std::string &device_name() const { return device_type_; }
  int device_id() const { return 0; }
  const std::vector<int64_t> &shape() const { return shape_; }
  rknn_tensor_type element_type() const { return type_; }
  const rknn_tensor_attr &attr() const { return attr_; }

  void apply_input_binding_attr(const rknn_tensor_attr &native_attr,
                                const rknn_tensor_attr &logical_attr,
                                const std::string &name) {
    const std::vector<int64_t> expected_shape = attr_shape_i64(logical_attr);
    require_shape_matches(shape_, expected_shape,
                          "OrtValue for input '" + name + "'");
    require_type_matches(type_, logical_attr.type,
                         "OrtValue for input '" + name + "'");
    require_native_attr_matches(attr_, native_attr,
                                "OrtValue for input '" + name + "'");
    require_buffer_size_for_attr(native_attr);
    attr_ = native_attr;
    shape_ = expected_shape;
    type_ = logical_attr.type;
  }

  void apply_output_binding_attr(const rknn_tensor_attr &native_attr,
                                 const rknn_tensor_attr &logical_attr,
                                 const std::string &name) {
    const std::vector<int64_t> expected_shape = attr_shape_i64(logical_attr);
    require_shape_matches(shape_, expected_shape,
                          "OrtValue for output '" + name + "'");
    require_type_matches(type_, logical_attr.type,
                         "OrtValue for output '" + name + "'");
    require_buffer_size_for_attr(native_attr);
    attr_ = native_attr;
    shape_ = expected_shape;
    type_ = logical_attr.type;
  }

  void require_update_source_compatible(const py::array &source) const {
    require_shape_matches(numpy_shape_i64(source), shape_,
                          "update_inplace source");
    py::dtype expected_dtype = rknn_tensor_type_to_numpy_dtype(type_);
    py::object dtype_matches =
        source.attr("dtype").attr("__eq__")(expected_dtype);
    if (!py::cast<bool>(dtype_matches)) {
      throw std::runtime_error(
          "update_inplace source element type mismatch: expected " +
          std::string(py::str(expected_dtype)) + ", got " +
          std::string(py::str(source.attr("dtype"))));
    }
  }

  void require_buffer_size_for_attr(const rknn_tensor_attr &native_attr) const {
    const uint32_t required = attr_alloc_bytes(native_attr);
    if (backing_size() < required) {
      throw std::runtime_error(
          "OrtValue buffer is too small for bound tensor: required " +
          std::to_string(required) + " bytes, got " +
          std::to_string(backing_size()) +
          " bytes (n_dims=" + std::to_string(native_attr.n_dims) +
          ", fmt=" + std::to_string(static_cast<int>(native_attr.fmt)) +
          ", size=" + std::to_string(native_attr.size) +
          ", size_with_stride=" +
          std::to_string(native_attr.size_with_stride) + ")");
    }
  }

  uintptr_t data_ptr() const {
    if (is_cpu()) {
      return reinterpret_cast<uintptr_t>(cpu_array_.data());
    }
    void *ptr = mutable_data_ptr();
    if (ptr == nullptr) {
      return 0;
    }
    return reinterpret_cast<uintptr_t>(ptr);
  }

  void sync_to_device() {
    if (!is_rknn()) {
      return;
    }
    for (auto &item : imports_) {
      sync_to_device(item.second.holder);
    }
  }

  void sync_from_device() {
    if (!is_rknn()) {
      return;
    }
    if (last_writer_) {
      sync_from_device(last_writer_);
      return;
    }
    if (!imports_.empty()) {
      sync_from_device(imports_.begin()->second.holder);
    }
  }

  rknn_tensor_mem *import_for(std::shared_ptr<RknnContextHolder> holder,
                              long long *import_us = nullptr) {
    if (!holder || holder->ctx == 0) {
      throw std::runtime_error("RKNN context is not available");
    }
    auto it = imports_.find(holder->ctx);
    if (it != imports_.end()) {
      return it->second.mem;
    }
    const auto start = std::chrono::steady_clock::now();
    rknn_tensor_mem *mem = nullptr;
    if (buffer_) {
      mem = rknn_create_mem_from_fd(
          holder->ctx, buffer_->fd(), buffer_->virt_addr(),
          static_cast<uint32_t>(buffer_->size()), buffer_->offset());
      if (mem == nullptr) {
        throw std::runtime_error("rknn_create_mem_from_fd failed");
      }
    } else {
      throw std::runtime_error(
          "OrtValue was allocated without dma-buf backing and cannot be "
          "imported into another RKNN context");
    }
    const auto end = std::chrono::steady_clock::now();
    if (import_us != nullptr) {
      *import_us +=
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count();
    }
    const rknn_context ctx_key = holder->ctx;
    ImportedMem imported;
    imported.holder = holder;
    imported.mem = mem;
    imports_[ctx_key] = std::move(imported);
    return imports_.at(ctx_key).mem;
  }

  void sync_to_device(std::shared_ptr<RknnContextHolder> holder) {
    if (!holder || holder->ctx == 0) {
      return;
    }
    rknn_tensor_mem *mem = import_for(holder);
    const int ret = rknn_mem_sync(holder->ctx, mem, RKNN_MEMORY_SYNC_TO_DEVICE);
    if (ret < 0) {
      throw std::runtime_error(format_rknn_error("rknn_mem_sync(TO_DEVICE)",
                                                 ret));
    }
  }

  void sync_from_device(std::shared_ptr<RknnContextHolder> holder) {
    if (!holder || holder->ctx == 0) {
      return;
    }
    rknn_tensor_mem *mem = import_for(holder);
    const int ret =
        rknn_mem_sync(holder->ctx, mem, RKNN_MEMORY_SYNC_FROM_DEVICE);
    if (ret < 0) {
      throw std::runtime_error(format_rknn_error("rknn_mem_sync(FROM_DEVICE)",
                                                 ret));
    }
  }

  void mark_device_written(std::shared_ptr<RknnContextHolder> holder) {
    last_writer_ = std::move(holder);
  }

  void update_inplace(py::array source) {
    if (is_cpu()) {
      py::array contiguous = py::array::ensure(source, py::array::c_style);
      if (!contiguous) {
        throw std::runtime_error("source must be a contiguous numpy array");
      }
      require_shape_matches(numpy_shape_i64(contiguous), shape_,
                            "update_inplace source");
      py::object dtype_matches =
          contiguous.attr("dtype").attr("__eq__")(cpu_array_.attr("dtype"));
      if (!py::cast<bool>(dtype_matches)) {
        throw std::runtime_error(
            "update_inplace source element type mismatch: expected " +
            std::string(py::str(cpu_array_.attr("dtype"))) + ", got " +
            std::string(py::str(contiguous.attr("dtype"))));
      }
      if (contiguous.nbytes() != cpu_array_.nbytes()) {
        throw std::runtime_error("source byte size does not match OrtValue");
      }
      std::memcpy(cpu_array_.mutable_data(), contiguous.data(),
                  static_cast<size_t>(contiguous.nbytes()));
      return;
    }
    void *dst_ptr = mutable_data_ptr();
    if (!is_rknn() || dst_ptr == nullptr) {
      throw std::runtime_error("OrtValue has no RKNN memory");
    }
    py::array contiguous = py::array::ensure(source, py::array::c_style);
    if (!contiguous) {
      throw std::runtime_error("source must be a contiguous numpy array");
    }
    require_update_source_compatible(contiguous);
    const size_t source_bytes = static_cast<size_t>(contiguous.nbytes());
    const size_t dense = attr_dense_bytes(attr_);
    const size_t alloc = attr_alloc_bytes(attr_);
    if (logical_shape_is_native_nhwc() &&
        should_copy_nhwc_to_nhwc(contiguous)) {
      copy_nhwc_to_nhwc(contiguous);
    } else if (should_transpose_nchw_to_nhwc(contiguous)) {
      copy_nchw_to_nhwc(contiguous);
    } else if (should_copy_nhwc_to_nhwc(contiguous)) {
      copy_nhwc_to_nhwc(contiguous);
    } else if (source_bytes == alloc) {
      std::memcpy(dst_ptr, contiguous.data(), source_bytes);
    } else if (source_bytes == dense && dense == alloc) {
      std::memcpy(dst_ptr, contiguous.data(), source_bytes);
    } else {
      require_dense_native_copy_supported(attr_, "update_inplace");
    }
    sync_to_device();
  }

  py::array numpy() {
    if (is_cpu()) {
      return cpu_array_;
    }
    void *src_ptr = mutable_data_ptr();
    if (!is_rknn() || src_ptr == nullptr) {
      throw std::runtime_error("OrtValue has no RKNN memory");
    }
    require_dense_native_copy_supported(attr_, "numpy");
    sync_from_device();
    py::dtype dtype = rknn_tensor_type_to_numpy_dtype(type_);
    py::array out(dtype, shape_);
    if (should_transpose_nhwc_to_nchw()) {
      copy_nhwc_to_nchw(out);
    } else {
      std::memcpy(out.mutable_data(), src_ptr,
                  static_cast<size_t>(attr_dense_bytes(attr_)));
    }
    return out;
  }

  py::dict memory_info() const {
    py::dict info;
    info["device_type"] = device_type_;
    info["device_id"] = 0;
    if (is_rknn()) {
      info["size"] = backing_size();
      info["fd"] = backing_fd();
      info["offset"] = backing_offset();
      info["virt_addr"] = reinterpret_cast<uintptr_t>(mutable_data_ptr());
      info["import_count"] = imports_.size();
      info["dmabuf_backed"] = static_cast<bool>(buffer_);
      info["attr_size"] = attr_.size;
      info["size_with_stride"] = attr_.size_with_stride;
      info["w_stride"] = attr_.w_stride;
      info["h_stride"] = attr_.h_stride;
      info["fmt"] = static_cast<int>(attr_.fmt);
      info["pass_through"] = static_cast<bool>(attr_.pass_through);
    }
    return info;
  }

private:
  bool should_transpose_nchw_to_nhwc(const py::array &source) const {
    if (attr_.fmt != RKNN_TENSOR_NHWC || attr_.n_dims != 4 ||
        source.ndim() != 4) {
      return false;
    }
    const ssize_t n = source.shape(0);
    const ssize_t c = source.shape(1);
    const ssize_t h = source.shape(2);
    const ssize_t w = source.shape(3);
    return n == static_cast<ssize_t>(attr_.dims[0]) &&
           h == static_cast<ssize_t>(attr_.dims[1]) &&
           w == static_cast<ssize_t>(attr_.dims[2]) &&
           c == static_cast<ssize_t>(attr_.dims[3]);
  }

  bool should_copy_nhwc_to_nhwc(const py::array &source) const {
    if (attr_.fmt != RKNN_TENSOR_NHWC || attr_.n_dims != 4 ||
        source.ndim() != 4) {
      return false;
    }
    return source.shape(0) == static_cast<ssize_t>(attr_.dims[0]) &&
           source.shape(1) == static_cast<ssize_t>(attr_.dims[1]) &&
           source.shape(2) == static_cast<ssize_t>(attr_.dims[2]) &&
           source.shape(3) == static_cast<ssize_t>(attr_.dims[3]);
  }

  bool logical_shape_is_native_nhwc() const {
    return attr_.fmt == RKNN_TENSOR_NHWC && attr_.n_dims == 4 &&
           shape_.size() == 4 &&
           shape_[0] == static_cast<int64_t>(attr_.dims[0]) &&
           shape_[1] == static_cast<int64_t>(attr_.dims[1]) &&
           shape_[2] == static_cast<int64_t>(attr_.dims[2]) &&
           shape_[3] == static_cast<int64_t>(attr_.dims[3]);
  }

  bool should_transpose_nhwc_to_nchw() const {
    if (attr_.fmt != RKNN_TENSOR_NHWC || attr_.n_dims != 4 ||
        shape_.size() != 4) {
      return false;
    }
    return shape_[0] == static_cast<int64_t>(attr_.dims[0]) &&
           shape_[2] == static_cast<int64_t>(attr_.dims[1]) &&
           shape_[3] == static_cast<int64_t>(attr_.dims[2]) &&
           shape_[1] == static_cast<int64_t>(attr_.dims[3]);
  }

  void copy_nchw_to_nhwc(const py::array &source) {
    const size_t elem_size = rknn_tensor_type_size(attr_.type);
    const size_t n = static_cast<size_t>(source.shape(0));
    const size_t c = static_cast<size_t>(source.shape(1));
    const size_t h = static_cast<size_t>(source.shape(2));
    const size_t w = static_cast<size_t>(source.shape(3));
    const size_t h_stride = attr_height_stride(attr_);
    const size_t w_stride = attr_width_stride(attr_);
    if (h_stride < h || w_stride < w) {
      throw std::runtime_error("NCHW to NHWC update_inplace has invalid RKNN stride");
    }
    const auto *src = static_cast<const uint8_t *>(source.data());
    auto *dst = static_cast<uint8_t *>(mutable_data_ptr());
    for (size_t ni = 0; ni < n; ++ni) {
      for (size_t hi = 0; hi < h; ++hi) {
        for (size_t wi = 0; wi < w; ++wi) {
          for (size_t ci = 0; ci < c; ++ci) {
            const size_t src_index = ((ni * c + ci) * h + hi) * w + wi;
            const size_t dst_index =
                ((ni * h_stride + hi) * w_stride + wi) * c + ci;
            std::memcpy(dst + dst_index * elem_size, src + src_index * elem_size,
                        elem_size);
          }
        }
      }
    }
  }

  void copy_nhwc_to_nhwc(const py::array &source) {
    const size_t elem_size = rknn_tensor_type_size(attr_.type);
    const size_t n = static_cast<size_t>(source.shape(0));
    const size_t h = static_cast<size_t>(source.shape(1));
    const size_t w = static_cast<size_t>(source.shape(2));
    const size_t c = static_cast<size_t>(source.shape(3));
    const size_t h_stride = attr_height_stride(attr_);
    const size_t w_stride = attr_width_stride(attr_);
    if (h_stride < h || w_stride < w) {
      throw std::runtime_error("NHWC update_inplace has invalid RKNN stride");
    }
    const auto *src = static_cast<const uint8_t *>(source.data());
    auto *dst = static_cast<uint8_t *>(mutable_data_ptr());
    if (h_stride == h && w_stride == w) {
      std::memcpy(dst, src, static_cast<size_t>(source.nbytes()));
      return;
    }
    for (size_t ni = 0; ni < n; ++ni) {
      for (size_t hi = 0; hi < h; ++hi) {
        for (size_t wi = 0; wi < w; ++wi) {
          const size_t src_index = ((ni * h + hi) * w + wi) * c;
          const size_t dst_index = ((ni * h_stride + hi) * w_stride + wi) * c;
          std::memcpy(dst + dst_index * elem_size,
                      src + src_index * elem_size, c * elem_size);
        }
      }
    }
  }

  void copy_nhwc_to_nchw(py::array &destination) {
    const size_t elem_size = rknn_tensor_type_size(attr_.type);
    const size_t n = static_cast<size_t>(shape_[0]);
    const size_t c = static_cast<size_t>(shape_[1]);
    const size_t h = static_cast<size_t>(shape_[2]);
    const size_t w = static_cast<size_t>(shape_[3]);
    const size_t h_stride = attr_height_stride(attr_);
    const size_t w_stride = attr_width_stride(attr_);
    if (h_stride < h || w_stride < w) {
      throw std::runtime_error("NHWC to NCHW numpy has invalid RKNN stride");
    }
    const auto *src = static_cast<const uint8_t *>(mutable_data_ptr());
    auto *dst = static_cast<uint8_t *>(destination.mutable_data());
    for (size_t ni = 0; ni < n; ++ni) {
      for (size_t ci = 0; ci < c; ++ci) {
        for (size_t hi = 0; hi < h; ++hi) {
          for (size_t wi = 0; wi < w; ++wi) {
            const size_t src_index =
                ((ni * h_stride + hi) * w_stride + wi) * c + ci;
            const size_t dst_index = ((ni * c + ci) * h + hi) * w + wi;
            std::memcpy(dst + dst_index * elem_size,
                        src + src_index * elem_size, elem_size);
          }
        }
      }
    }
  }

  void *mutable_data_ptr() const {
    if (buffer_) {
      return buffer_->virt_addr();
    }
    if (!imports_.empty() && imports_.begin()->second.mem != nullptr) {
      return imports_.begin()->second.mem->virt_addr;
    }
    return nullptr;
  }

  size_t backing_size() const {
    if (buffer_) {
      return buffer_->size();
    }
    if (!imports_.empty() && imports_.begin()->second.mem != nullptr) {
      return imports_.begin()->second.mem->size;
    }
    return 0;
  }

  int backing_fd() const {
    if (buffer_) {
      return buffer_->fd();
    }
    if (!imports_.empty() && imports_.begin()->second.mem != nullptr) {
      return imports_.begin()->second.mem->fd;
    }
    return -1;
  }

  int32_t backing_offset() const {
    if (buffer_) {
      return buffer_->offset();
    }
    if (!imports_.empty() && imports_.begin()->second.mem != nullptr) {
      return imports_.begin()->second.mem->offset;
    }
    return 0;
  }

  void create_fallback_mem(std::shared_ptr<RknnContextHolder> holder,
                           uint32_t bytes) {
    rknn_tensor_mem *mem = rknn_create_mem(holder->ctx, bytes);
    if (mem == nullptr) {
      throw std::runtime_error("rknn_create_mem failed");
    }
    ImportedMem imported;
    imported.holder = holder;
    imported.mem = mem;
    imports_[holder->ctx] = std::move(imported);
  }

  std::string device_type_;
  py::array cpu_array_;
  struct ImportedMem {
    std::shared_ptr<RknnContextHolder> holder;
    rknn_tensor_mem *mem = nullptr;
  };
  std::shared_ptr<SharedDmaBuffer> buffer_;
  std::map<rknn_context, ImportedMem> imports_;
  std::shared_ptr<RknnContextHolder> last_writer_;
  rknn_tensor_attr attr_ = {};
  std::vector<int64_t> shape_;
  rknn_tensor_type type_ = RKNN_TENSOR_TYPE_MAX;
};

class ZeroCopyEzRknn : public std::enable_shared_from_this<ZeroCopyEzRknn> {
public:
  explicit ZeroCopyEzRknn(std::shared_ptr<RknnContextHolder> holder,
                          ZeroCopyInputLayout input_layout =
                              ZeroCopyInputLayout::NCHW)
      : holder_(std::move(holder)), input_layout_(input_layout),
        print_perf_(read_perf_env_enabled()) {
    init_from_context();
  }

  std::shared_ptr<RknnContextHolder> holder() const { return holder_; }
  const std::vector<rknn_tensor_attr> &input_attrs() const {
    return input_attrs_;
  }
  const std::vector<rknn_tensor_attr> &output_attrs() const {
    return output_attrs_;
  }
  const std::vector<rknn_tensor_attr> &native_input_attrs() const {
    return native_input_attrs_;
  }
  const std::vector<rknn_tensor_attr> &native_output_attrs() const {
    return native_output_attrs_;
  }

  std::vector<std::string> input_names() const { return input_names_; }
  std::vector<std::string> output_names() const { return output_names_; }

  size_t input_index(const std::string &name) const {
    return lookup_name(input_name_to_index_, name, "input");
  }

  size_t output_index(const std::string &name) const {
    return lookup_name(output_name_to_index_, name, "output");
  }

  const rknn_tensor_attr &native_input_attr(const std::string &name) const {
    return native_input_attrs_.at(input_index(name));
  }

  const rknn_tensor_attr &native_output_attr(const std::string &name) const {
    return native_output_attrs_.at(output_index(name));
  }

  const rknn_tensor_attr &logical_input_attr(const std::string &name) const {
    return input_attrs_.at(input_index(name));
  }

  const rknn_tensor_attr &logical_output_attr(const std::string &name) const {
    return output_attrs_.at(output_index(name));
  }

  std::shared_ptr<RknnOrtValue>
  create_value_for_io(const std::string &name, const std::string &io_kind,
                      rknn_tensor_type type, uint64_t alloc_flags,
                      const std::optional<std::vector<int64_t>> &requested_shape =
                          std::nullopt) {
    const bool is_input = io_kind == "input";
    const bool is_output = io_kind == "output";
    if (!is_input && !is_output) {
      throw std::runtime_error("io_kind must be 'input' or 'output'");
    }
    rknn_tensor_attr attr =
        is_input ? native_input_attr(name) : native_output_attr(name);
    rknn_tensor_attr logical =
        is_input ? logical_input_attr(name) : logical_output_attr(name);
    const std::string context = "OrtValue for " + io_kind + " '" + name + "'";
    const std::vector<int64_t> expected_shape = attr_shape_i64(logical);
    if (requested_shape.has_value()) {
      require_shape_matches(requested_shape.value(), expected_shape, context);
    }
    if (type != RKNN_TENSOR_TYPE_MAX) {
      require_type_matches(type, logical.type, context);
    }
    return RknnOrtValue::create_rknn(holder_, attr, attr_shape_i64(logical),
                                     logical.type, alloc_flags);
  }

  std::shared_ptr<RknnOrtValue>
  create_value_from_fd(const std::string &name, const std::string &io_kind,
                       rknn_tensor_type type, int32_t fd, void *virt_addr,
                       uint32_t size, int32_t offset,
                       const std::optional<std::vector<int64_t>> &requested_shape =
                           std::nullopt) {
    const bool is_input = io_kind == "input";
    const bool is_output = io_kind == "output";
    if (!is_input && !is_output) {
      throw std::runtime_error("io_kind must be 'input' or 'output'");
    }
    rknn_tensor_attr attr =
        is_input ? native_input_attr(name) : native_output_attr(name);
    rknn_tensor_attr logical =
        is_input ? logical_input_attr(name) : logical_output_attr(name);
    const std::string context = "OrtValue from dmabuf for " + io_kind + " '" +
                                name + "'";
    const std::vector<int64_t> expected_shape = attr_shape_i64(logical);
    if (requested_shape.has_value()) {
      require_shape_matches(requested_shape.value(), expected_shape, context);
    }
    require_type_matches(type, logical.type, context);
    return RknnOrtValue::from_fd(holder_, fd, virt_addr, size, offset, attr,
                                 attr_shape_i64(logical), logical.type);
  }

  void run_with_iobinding(RknnIoBinding &binding);

private:
  struct PerfStats {
    long long import_us = 0;
    long long set_input_us = 0;
    long long infer_us = 0;
    long long copy_output_us = 0;
  };

  static bool parse_env_flag_value(const char *value) {
    if (value == nullptr) {
      return false;
    }
    std::string text(value);
    for (char &ch : text) {
      ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return text == "1" || text == "true" || text == "on" || text == "yes";
  }

  static bool read_perf_env_enabled() {
    return parse_env_flag_value(std::getenv("ZTU_EZRKNN_ASYNC_PRINT_PERF"));
  }

  void log_perf_line(size_t task_id, const PerfStats &stats) {
    if (!print_perf_) {
      return;
    }
    const long long total_us = stats.import_us + stats.set_input_us +
                               stats.infer_us + stats.copy_output_us;
    std::cerr << "[ztu_ez_rknn_async_perf]"
              << " task_id=" << task_id << " thread_id=-1"
              << " core_id=-1"
              << " import_us=" << stats.import_us
              << " set_input_us=" << stats.set_input_us
              << " infer_us=" << stats.infer_us
              << " copy_output_us=" << stats.copy_output_us
              << " total_us=" << total_us << " mode=zero_copy" << std::endl;
  }

  static size_t lookup_name(const std::map<std::string, size_t> &lookup,
                            const std::string &name, const char *kind) {
    auto it = lookup.find(name);
    if (it == lookup.end()) {
      throw std::runtime_error("Unknown " + std::string(kind) + " name: " +
                               name);
    }
    return it->second;
  }

  void init_from_context() {
    if (!holder_ || holder_->ctx == 0) {
      throw std::runtime_error("RKNN context is not available");
    }

    int ret = rknn_query(holder_->ctx, RKNN_QUERY_IN_OUT_NUM, &io_num_,
                         sizeof(io_num_));
    if (ret < 0) {
      throw std::runtime_error(
          format_rknn_error("rknn_query(RKNN_QUERY_IN_OUT_NUM)", ret));
    }

    input_attrs_.resize(io_num_.n_input);
    native_input_attrs_.resize(io_num_.n_input);
    for (uint32_t i = 0; i < io_num_.n_input; ++i) {
      rknn_tensor_attr queried_input_attr = {};
      queried_input_attr.index = i;
      ret = rknn_query(holder_->ctx, RKNN_QUERY_INPUT_ATTR, &queried_input_attr,
                       sizeof(rknn_tensor_attr));
      if (ret < 0) {
        throw std::runtime_error(format_rknn_error_for_index(
            "rknn_query(RKNN_QUERY_INPUT_ATTR)", "input attr", i, ret));
      }
      input_attrs_[i] =
          make_user_input_attr_for_layout(queried_input_attr, input_layout_);
      input_attrs_[i].index = i;
      native_input_attrs_[i].index = i;
      ret = rknn_query(holder_->ctx, RKNN_QUERY_NATIVE_INPUT_ATTR,
                       &native_input_attrs_[i], sizeof(rknn_tensor_attr));
      if (ret < 0 || input_attrs_[i].n_dims != 4) {   //FIXME: zt: 5d tensor returned a completely wrong native shape, this looks like a vendor bug
        native_input_attrs_[i] = input_attrs_[i];
      } else if (supports_native_nhwc_query_type(input_attrs_[i].type)) {
        rknn_tensor_attr nhwc_attr = {};
        nhwc_attr.index = i;
        const int nhwc_ret =
            rknn_query(holder_->ctx, RKNN_QUERY_NATIVE_NHWC_INPUT_ATTR,
                       &nhwc_attr, sizeof(rknn_tensor_attr));
        if (nhwc_ret >= 0 && nhwc_attr.n_dims == 4) {
          native_input_attrs_[i] = nhwc_attr;
        }
        if (can_bind_nc1hwc2_input_as_nhwc(native_input_attrs_[i],
                                           input_attrs_[i])) {
          native_input_attrs_[i] = make_nhwc_input_attr_from_logical(
              native_input_attrs_[i], input_attrs_[i]);
        }
      }
    }

    output_attrs_.resize(io_num_.n_output);
    native_output_attrs_.resize(io_num_.n_output);
    for (uint32_t i = 0; i < io_num_.n_output; ++i) {
      output_attrs_[i].index = i;
      ret = rknn_query(holder_->ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attrs_[i],
                       sizeof(rknn_tensor_attr));
      if (ret < 0) {
        throw std::runtime_error(format_rknn_error_for_index(
            "rknn_query(RKNN_QUERY_OUTPUT_ATTR)", "output attr", i, ret));
      }
      native_output_attrs_[i].index = i;
      ret = rknn_query(holder_->ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR,
                       &native_output_attrs_[i], sizeof(rknn_tensor_attr));
      if (ret < 0 || output_attrs_[i].n_dims != 4) {
        native_output_attrs_[i] = output_attrs_[i];
      } else if (supports_native_nhwc_query_type(output_attrs_[i].type)) {
        rknn_tensor_attr nhwc_attr = {};
        nhwc_attr.index = i;
        const int nhwc_ret =
            rknn_query(holder_->ctx, RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR,
                       &nhwc_attr, sizeof(rknn_tensor_attr));
        if (nhwc_ret >= 0 && nhwc_attr.n_dims == 4) {
          native_output_attrs_[i] = nhwc_attr;
        }
      }
    }

    input_names_.reserve(input_attrs_.size());
    for (size_t i = 0; i < input_attrs_.size(); ++i) {
      const auto &attr = input_attrs_[i];
      std::string name =
          attr.name[0] ? std::string(attr.name) : "input_" + std::to_string(i);
      input_name_to_index_[name] = i;
      input_names_.push_back(name);
    }

    output_names_.reserve(output_attrs_.size());
    for (size_t i = 0; i < output_attrs_.size(); ++i) {
      const auto &attr = output_attrs_[i];
      std::string name =
          attr.name[0] ? std::string(attr.name) : "output_" + std::to_string(i);
      output_name_to_index_[name] = i;
      output_names_.push_back(name);
    }
  }

  std::shared_ptr<RknnContextHolder> holder_;
  ZeroCopyInputLayout input_layout_ = ZeroCopyInputLayout::NCHW;
  rknn_input_output_num io_num_ = {};
  std::vector<rknn_tensor_attr> input_attrs_;
  std::vector<rknn_tensor_attr> output_attrs_;
  std::vector<rknn_tensor_attr> native_input_attrs_;
  std::vector<rknn_tensor_attr> native_output_attrs_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::map<std::string, size_t> input_name_to_index_;
  std::map<std::string, size_t> output_name_to_index_;
  std::atomic<size_t> task_counter_{0};
  bool print_perf_ = false;
};

class RknnIoBinding {
public:
  explicit RknnIoBinding(std::shared_ptr<ZeroCopyEzRknn> session)
      : session_(std::move(session)) {
    if (!session_) {
      throw std::runtime_error("SessionIOBinding requires a valid session");
    }
  }

  void bind_cpu_input(const std::string &name, py::array array) {
    auto value =
        session_->create_value_for_io(name, "input", RKNN_TENSOR_TYPE_MAX,
                                      RKNN_FLAG_MEMORY_FLAGS_DEFAULT);
    value->update_inplace(array);
    bind_ortvalue_input(name, value);
  }

  void bind_ortvalue_input(const std::string &name,
                           std::shared_ptr<RknnOrtValue> value) {
    if (!value || !value->is_rknn()) {
      throw std::runtime_error("bind_ortvalue_input requires an RKNN OrtValue");
    }
    const size_t idx = session_->input_index(name);
    value->apply_input_binding_attr(session_->native_input_attrs().at(idx),
                                    session_->input_attrs().at(idx), name);
    bound_inputs_[idx] = std::move(value);
  }

  void bind_ortvalue_output(const std::string &name,
                            std::shared_ptr<RknnOrtValue> value) {
    if (!value || !value->is_rknn()) {
      throw std::runtime_error("bind_ortvalue_output requires an RKNN OrtValue");
    }
    const size_t idx = session_->output_index(name);
    value->apply_output_binding_attr(session_->native_output_attrs().at(idx),
                                     session_->output_attrs().at(idx), name);
    bound_outputs_[idx] = std::move(value);
    remember_output_order(idx);
  }

  void bind_output(const std::string &name, const std::string &device_type,
                   int device_id, std::optional<rknn_tensor_type> element_type,
                   const std::optional<std::vector<int64_t>> &shape,
                   std::optional<uint64_t> buffer_ptr, uint64_t alloc_flags) {
    (void)device_id;
    (void)shape;
    if (buffer_ptr.has_value()) {
      throw std::runtime_error(
          "bind_output with raw buffer_ptr is not implemented; use "
          "bind_ortvalue_output or let bind_output allocate an OrtValue");
    }
    if (device_type != "cpu" && device_type != "rknpu2") {
      throw std::runtime_error("device_type must be 'cpu' or 'rknpu2'");
    }
    auto value = session_->create_value_for_io(
        name, "output", element_type.value_or(RKNN_TENSOR_TYPE_MAX),
        alloc_flags, shape);
    const size_t idx = session_->output_index(name);
    bound_outputs_[idx] = std::move(value);
    if (device_type == "cpu") {
      cpu_output_indices_[idx] = true;
    }
    remember_output_order(idx);
  }

  std::vector<std::shared_ptr<RknnOrtValue>> get_outputs() const {
    std::vector<std::shared_ptr<RknnOrtValue>> outputs;
    outputs.reserve(output_order_.size());
    for (size_t idx : output_order_) {
      auto it = bound_outputs_.find(idx);
      if (it != bound_outputs_.end()) {
        outputs.push_back(it->second);
      }
    }
    return outputs;
  }

  py::list copy_outputs_to_cpu() {
    py::list result;
    for (size_t idx : output_order_) {
      auto it = bound_outputs_.find(idx);
      if (it == bound_outputs_.end()) {
        continue;
      }
      result.append(it->second->numpy());
    }
    return result;
  }

  void clear_binding_inputs() { bound_inputs_.clear(); }

  void clear_binding_outputs() {
    bound_outputs_.clear();
    cpu_output_indices_.clear();
    output_order_.clear();
  }

  void synchronize_inputs() {
    for (auto &item : bound_inputs_) {
      item.second->sync_to_device();
    }
  }

  void synchronize_outputs() {
    for (auto &item : bound_outputs_) {
      item.second->sync_from_device();
    }
  }

private:
  friend class ZeroCopyEzRknn;

  void remember_output_order(size_t idx) {
    if (std::find(output_order_.begin(), output_order_.end(), idx) ==
        output_order_.end()) {
      output_order_.push_back(idx);
    }
  }

  std::shared_ptr<ZeroCopyEzRknn> session_;
  std::map<size_t, std::shared_ptr<RknnOrtValue>> bound_inputs_;
  std::map<size_t, std::shared_ptr<RknnOrtValue>> bound_outputs_;
  std::map<size_t, bool> cpu_output_indices_;
  std::vector<size_t> output_order_;
};

inline void ZeroCopyEzRknn::run_with_iobinding(RknnIoBinding &binding) {
  if (binding.session_.get() != this) {
    throw std::runtime_error("IOBinding belongs to a different session");
  }
  if (binding.bound_inputs_.size() != input_attrs_.size()) {
    throw std::runtime_error("All model inputs must be bound before "
                             "run_with_iobinding");
  }
  if (binding.bound_outputs_.size() != output_attrs_.size()) {
    throw std::runtime_error("All model outputs must be bound before "
                             "run_with_iobinding");
  }

  const size_t task_id = task_counter_++;
  PerfStats perf_stats;
  {
    const auto set_input_start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < native_input_attrs_.size(); ++i) {
      auto it = binding.bound_inputs_.find(i);
      if (it == binding.bound_inputs_.end()) {
        throw std::runtime_error("Input is not bound: " + input_names_[i]);
      }
      rknn_tensor_mem *mem = it->second->import_for(holder_, &perf_stats.import_us);
      it->second->sync_to_device(holder_);
      if (mem == nullptr ||
          mem->size < attr_alloc_bytes(native_input_attrs_[i])) {
        throw std::runtime_error("Bound input OrtValue is too small for input: " +
                                 input_names_[i]);
      }
      rknn_tensor_attr attr = native_input_attrs_[i];
      attr.index = static_cast<uint32_t>(i);
      const int ret = rknn_set_io_mem(holder_->ctx, mem, &attr);  //TODO: zt: do we need to call this every time?
      if (ret < 0) {
        throw std::runtime_error(format_rknn_error_for_index(
            "rknn_set_io_mem", "input", static_cast<uint32_t>(i), ret));
      }
    }

    for (size_t i = 0; i < native_output_attrs_.size(); ++i) {
      auto it = binding.bound_outputs_.find(i);
      if (it == binding.bound_outputs_.end()) {
        throw std::runtime_error("Output is not bound: " + output_names_[i]);
      }
      rknn_tensor_mem *mem = it->second->import_for(holder_, &perf_stats.import_us);
      if (mem == nullptr ||
          mem->size < attr_alloc_bytes(native_output_attrs_[i])) {
        throw std::runtime_error(
            "Bound output OrtValue is too small for output: " +
            output_names_[i]);
      }
      rknn_tensor_attr attr = native_output_attrs_[i];
      attr.index = static_cast<uint32_t>(i);
      const int ret = rknn_set_io_mem(holder_->ctx, mem, &attr);
      if (ret < 0) {
        throw std::runtime_error(format_rknn_error_for_index(
            "rknn_set_io_mem", "output", static_cast<uint32_t>(i), ret));
      }
    }
    const auto set_input_end = std::chrono::steady_clock::now();
    perf_stats.set_input_us =
        std::chrono::duration_cast<std::chrono::microseconds>(set_input_end -
                                                              set_input_start)
            .count();
  }

  const auto inference_start = std::chrono::steady_clock::now();
  const int ret = rknn_run(holder_->ctx, nullptr);
  const auto inference_end = std::chrono::steady_clock::now();
  perf_stats.infer_us =
      std::chrono::duration_cast<std::chrono::microseconds>(inference_end -
                                                            inference_start)
          .count();
  log_perf_line(task_id, perf_stats);
  if (ret < 0) {
    throw std::runtime_error(format_rknn_error("rknn_run", ret));
  }
  for (auto &item : binding.bound_outputs_) {
    item.second->mark_device_written(holder_);
  }
}

} // namespace rk
} // namespace ztu
