#pragma once
#include "rknn_api.h"

// 添加条件编译，避免与BoTSORT中的rknn_dtype.hpp冲突
#ifndef BOTSORT_RKNN_DTYPE_INCLUDED
#define BOTSORT_RKNN_DTYPE_INCLUDED

// 添加 RknnDtype 模板结构体，与 ez_rknn.hpp 保持一致
template <typename T> struct RknnDtype {};

template <typename T> struct RknnDtype<const T> : RknnDtype<T> {};

template <> struct RknnDtype<float> {
  static constexpr rknn_tensor_type type = RKNN_TENSOR_FLOAT32;
};

template <> struct RknnDtype<__fp16> {
  static constexpr rknn_tensor_type type = RKNN_TENSOR_FLOAT16;
};

template <> struct RknnDtype<int8_t> {
  static constexpr rknn_tensor_type type = RKNN_TENSOR_INT8;
};

template <> struct RknnDtype<uint8_t> {
  static constexpr rknn_tensor_type type = RKNN_TENSOR_UINT8;
};

template <> struct RknnDtype<int16_t> {
  static constexpr rknn_tensor_type type = RKNN_TENSOR_INT16;
};

template <> struct RknnDtype<uint16_t> {
  static constexpr rknn_tensor_type type = RKNN_TENSOR_UINT16;
};

template <> struct RknnDtype<int32_t> {
  static constexpr rknn_tensor_type type = RKNN_TENSOR_INT32;
};

template <> struct RknnDtype<uint32_t> {
  static constexpr rknn_tensor_type type = RKNN_TENSOR_UINT32;
};

template <> struct RknnDtype<int64_t> {
  static constexpr rknn_tensor_type type = RKNN_TENSOR_INT64;
};

template <> struct RknnDtype<bool> {
  static constexpr rknn_tensor_type type = RKNN_TENSOR_BOOL;
};

#endif // BOTSORT_RKNN_DTYPE_INCLUDED