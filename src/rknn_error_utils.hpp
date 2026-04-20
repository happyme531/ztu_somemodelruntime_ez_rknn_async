#pragma once

#include "rknn_api.h"

#include <cstddef>
#include <string>

namespace ztu {
namespace rk {

inline const char *rknn_status_name(int code) {
  switch (code) {
  case RKNN_SUCC:
    return "RKNN_SUCC";
  case RKNN_ERR_FAIL:
    return "RKNN_ERR_FAIL";
  case RKNN_ERR_TIMEOUT:
    return "RKNN_ERR_TIMEOUT";
  case RKNN_ERR_DEVICE_UNAVAILABLE:
    return "RKNN_ERR_DEVICE_UNAVAILABLE";
  case RKNN_ERR_MALLOC_FAIL:
    return "RKNN_ERR_MALLOC_FAIL";
  case RKNN_ERR_PARAM_INVALID:
    return "RKNN_ERR_PARAM_INVALID";
  case RKNN_ERR_MODEL_INVALID:
    return "RKNN_ERR_MODEL_INVALID";
  case RKNN_ERR_CTX_INVALID:
    return "RKNN_ERR_CTX_INVALID";
  case RKNN_ERR_INPUT_INVALID:
    return "RKNN_ERR_INPUT_INVALID";
  case RKNN_ERR_OUTPUT_INVALID:
    return "RKNN_ERR_OUTPUT_INVALID";
  case RKNN_ERR_DEVICE_UNMATCH:
    return "RKNN_ERR_DEVICE_UNMATCH";
  case RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL:
    return "RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL";
  case RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION:
    return "RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION";
  case RKNN_ERR_TARGET_PLATFORM_UNMATCH:
    return "RKNN_ERR_TARGET_PLATFORM_UNMATCH";
  default:
    return "RKNN_ERR_UNKNOWN";
  }
}

inline const char *rknn_status_description(int code) {
  switch (code) {
  case RKNN_SUCC:
    return "execute succeed.";
  case RKNN_ERR_FAIL:
    return "execute failed.";
  case RKNN_ERR_TIMEOUT:
    return "execute timeout.";
  case RKNN_ERR_DEVICE_UNAVAILABLE:
    return "device is unavailable.";
  case RKNN_ERR_MALLOC_FAIL:
    return "memory malloc fail.";
  case RKNN_ERR_PARAM_INVALID:
    return "parameter is invalid.";
  case RKNN_ERR_MODEL_INVALID:
    return "model is invalid.";
  case RKNN_ERR_CTX_INVALID:
    return "context is invalid.";
  case RKNN_ERR_INPUT_INVALID:
    return "input is invalid.";
  case RKNN_ERR_OUTPUT_INVALID:
    return "output is invalid.";
  case RKNN_ERR_DEVICE_UNMATCH:
    return "the device is unmatch, please update rknn sdk and npu driver/firmware.";
  case RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL:
    return "pre_compile model is not compatible with current driver.";
  case RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION:
    return "model optimization level is not compatible with current driver.";
  case RKNN_ERR_TARGET_PLATFORM_UNMATCH:
    return "model target platform is not compatible with current platform.";
  default:
    return "unknown RKNN error.";
  }
}

inline std::string format_rknn_status(int code) {
  return std::string(rknn_status_name(code)) + " (" + std::to_string(code) +
         "): " + rknn_status_description(code);
}

inline std::string format_rknn_error(const std::string &api, int code) {
  return api + " failed: " + format_rknn_status(code);
}

inline std::string format_rknn_error_for_index(const std::string &api,
                                               const std::string &kind,
                                               size_t index, int code) {
  return api + " failed for " + kind + " index " + std::to_string(index) +
         ": " + format_rknn_status(code);
}

} // namespace rk
} // namespace ztu
