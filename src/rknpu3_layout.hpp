#pragma once

#include "rknpu3_abi.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ztu::rk {

inline size_t checked_rknpu3_size_mul(size_t left, size_t right,
                                      const char *context) {
  if (right != 0 && left > std::numeric_limits<size_t>::max() / right)
    throw std::runtime_error(context);
  return left * right;
}

inline size_t checked_rknpu3_size_add(size_t left, size_t right,
                                      const char *context) {
  if (left > std::numeric_limits<size_t>::max() - right)
    throw std::runtime_error(context);
  return left + right;
}

inline void
copy_nchw_to_rknpu3_nc1hwc2(const void *source, size_t source_bytes,
                            const std::vector<int64_t> &source_shape,
                            void *destination, size_t destination_bytes,
                            size_t element_size,
                            const rknpu3_abi::TensorAttr &native_attr) {
  using namespace rknpu3_abi;

  if (native_attr.layout != TensorLayout::Nc1hwc2 || native_attr.n_dims != 5)
    throw std::runtime_error(
        "RKNPU3 NCHW upload requires a 5D NC1HWC2 native tensor");
  if (source_shape.size() != 4)
    throw std::runtime_error("RKNPU3 NCHW upload requires a 4D input");
  if (element_size == 0)
    throw std::runtime_error("RKNPU3 NCHW upload has invalid element size");
  if ((source == nullptr && source_bytes != 0) || destination == nullptr)
    throw std::runtime_error("RKNPU3 NCHW upload has a null buffer");

  std::array<size_t, 4> logical = {};
  size_t logical_elements = 1;
  for (size_t i = 0; i < logical.size(); ++i) {
    if (source_shape[i] < 0)
      throw std::runtime_error("RKNPU3 NCHW input has a negative dimension");
    logical[i] = static_cast<size_t>(source_shape[i]);
    logical_elements = checked_rknpu3_size_mul(
        logical_elements, logical[i], "RKNPU3 NCHW input is too large");
  }
  const size_t required_source_bytes = checked_rknpu3_size_mul(
      logical_elements, element_size, "RKNPU3 NCHW input is too large");
  if (source_bytes != required_source_bytes)
    throw std::runtime_error("RKNPU3 NCHW input byte size does not match");

  const size_t native_n = native_attr.shape[0];
  const size_t native_c1 = native_attr.shape[1];
  const size_t native_h = native_attr.shape[2];
  const size_t native_w = native_attr.shape[3];
  const size_t native_c2 = native_attr.shape[4];
  if (native_n == 0 || native_c1 == 0 || native_h == 0 || native_w == 0 ||
      native_c2 == 0)
    throw std::runtime_error("RKNPU3 NC1HWC2 tensor has an empty dimension");

  // TODO(rknpu3-runtime): Replace these capacity-only checks with the exact
  // logical shape returned by RKNN3_QUERY_ORIGIN_INPUT_ATTR once the RKCP
  // runtime implements that public query correctly.
  const size_t native_channels = checked_rknpu3_size_mul(
      native_c1, native_c2, "RKNPU3 native channel count is too large");
  if (logical[0] > native_n || logical[1] > native_channels ||
      logical[2] > native_h || logical[3] > native_w)
    throw std::runtime_error(
        "RKNPU3 NCHW input does not fit the native NC1HWC2 tensor");

  std::array<size_t, 5> strides = {};
  if (native_attr.n_stride >= strides.size()) {
    for (size_t i = 0; i < strides.size(); ++i)
      strides[i] = static_cast<size_t>(native_attr.stride[i]);
  } else {
    strides[4] = 1;
    for (size_t reverse = 1; reverse < strides.size(); ++reverse) {
      const size_t i = strides.size() - 1 - reverse;
      strides[i] =
          checked_rknpu3_size_mul(strides[i + 1], native_attr.shape[i + 1],
                                  "RKNPU3 NC1HWC2 stride is too large");
    }
  }
  for (size_t stride : strides)
    if (stride == 0)
      throw std::runtime_error("RKNPU3 NC1HWC2 tensor has a zero stride");

  size_t last_element = 0;
  if (logical_elements != 0) {
    last_element = checked_rknpu3_size_mul(logical[0] - 1, strides[0],
                                           "RKNPU3 destination is too large");
    last_element = checked_rknpu3_size_add(
        last_element,
        checked_rknpu3_size_mul((logical[1] - 1) / native_c2, strides[1],
                                "RKNPU3 destination is too large"),
        "RKNPU3 destination is too large");
    last_element = checked_rknpu3_size_add(
        last_element,
        checked_rknpu3_size_mul(logical[2] - 1, strides[2],
                                "RKNPU3 destination is too large"),
        "RKNPU3 destination is too large");
    last_element = checked_rknpu3_size_add(
        last_element,
        checked_rknpu3_size_mul(logical[3] - 1, strides[3],
                                "RKNPU3 destination is too large"),
        "RKNPU3 destination is too large");
    last_element = checked_rknpu3_size_add(
        last_element,
        checked_rknpu3_size_mul((logical[1] - 1) % native_c2, strides[4],
                                "RKNPU3 destination is too large"),
        "RKNPU3 destination is too large");
  }
  const size_t destination_elements =
      logical_elements == 0
          ? 0
          : checked_rknpu3_size_add(last_element, 1,
                                    "RKNPU3 destination is too large");
  const size_t required_destination_bytes = checked_rknpu3_size_mul(
      destination_elements, element_size, "RKNPU3 destination is too large");
  if (required_destination_bytes > destination_bytes)
    throw std::runtime_error("RKNPU3 NC1HWC2 destination is too small");

  std::memset(destination, 0, destination_bytes);
  const auto *source_bytes_ptr = static_cast<const uint8_t *>(source);
  auto *destination_bytes_ptr = static_cast<uint8_t *>(destination);
  const size_t channels = logical[1];
  const size_t height = logical[2];
  const size_t width = logical[3];
  for (size_t n = 0; n < logical[0]; ++n) {
    for (size_t c1 = 0; c1 < native_c1; ++c1) {
      for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
          for (size_t c2 = 0; c2 < native_c2; ++c2) {
            const size_t channel = c1 * native_c2 + c2;
            if (channel >= channels)
              continue;
            const size_t source_index =
                ((n * channels + channel) * height + h) * width + w;
            const size_t destination_index = n * strides[0] + c1 * strides[1] +
                                             h * strides[2] + w * strides[3] +
                                             c2 * strides[4];
            std::memcpy(
                destination_bytes_ptr + destination_index * element_size,
                source_bytes_ptr + source_index * element_size, element_size);
          }
        }
      }
    }
  }
}

} // namespace ztu::rk
