#pragma once

#include <cstdint>

// Minimal private ABI declarations for the RKNN3 runtime functions used by this
// project. The runtime is loaded with dlopen, so RKNPU2-only builds do not need
// an RKNN3 SDK installation at build or import time.
namespace ztu::rk::rknpu3_abi {

constexpr int kSuccess = 0;
constexpr int kErrorFail = -1;
constexpr int kWarningCoreUnused = -100;

constexpr uint32_t kMaxDims = 16;
constexpr uint32_t kMaxStrideDims = kMaxDims + 1;
constexpr uint32_t kMaxNameLen = 256;

#if defined(__arm__) || (defined(__riscv) && __riscv_xlen == 32)
using Context = uint32_t;
#else
using Context = uint64_t;
#endif

enum class QueryCommand : int {
  InOutNum = 0,
  InputAttr = 1,
  OutputAttr = 2,
  SdkVersion = 5,
  CustomString = 7,
  OriginInputAttr = 8,
  OriginOutputAttr = 9,
  CoreNumber = 13,
};

enum class TensorType : int {
  Float32 = 0,
  Float16 = 1,
  Int8 = 2,
  Uint8 = 3,
  Int16 = 4,
  Uint16 = 5,
  Int32 = 6,
  Uint32 = 7,
  Int64 = 8,
  Uint64 = 9,
  Bool = 10,
  Int4 = 11,
  Float8E4M3Fn = 12,
  BFloat16 = 13,
  Float8E8M0 = 14,
  Float4E2M1 = 15,
  Max = 16,
};

enum class TensorLayout : int {
  Undefined = 0,
  Nchw = 1,
  Nhwc = 2,
  Nc1hwc2 = 3,
  Chwn = 4,
  Hwio = 5,
  Oihw = 6,
  O1i1hwi2o2 = 7,
  Max = 8,
};

enum class TensorQuantType : int {
  None = 0,
  PerLayerSymmetric = 1,
  PerLayerAsymmetric = 2,
  PerChannelSymmetric = 3,
  PerChannelAsymmetric = 4,
  PerGroupSymmetric = 5,
  PerGroupAsymmetric = 6,
  Max = 7,
};

enum class MemoryType : int {
  NpuDram = 0,
  ExternalDdr = 1,
};

enum class MemoryAllocFlags : int {
  Default = 0,
  Cacheable = 1 << 0,
  NonCacheable = 1 << 1,
};

enum class MemorySyncMode : int {
  ToDevice = 0x1,
  FromDevice = 0x2,
  Bidirectional = 0x3,
};

struct InputOutputNum {
  uint32_t n_input;
  uint32_t n_output;
};

struct QuantInfo {
  float scale;
  int32_t zero_point;
};

struct TensorAttr {
  uint32_t index;
  char name[kMaxNameLen];
  uint32_t n_dims;
  uint32_t shape[kMaxDims];
  uint64_t aligned_size;
  uint32_t n_stride;
  uint64_t stride[kMaxStrideDims];
  uint32_t n_elems;
  TensorType dtype;
  TensorLayout layout;
  TensorQuantType qnt_type;
  QuantInfo qnt_info;
  int32_t core_id;
};

struct TensorMem {
  void *virt_addr;
  uint64_t phys_addr;
  int32_t fd;
  uint64_t buffer_id;
  uint64_t offset;
  uint64_t size;
  uint64_t flags;
  int32_t core_id;
  void *priv_data;
  MemoryType mem_type;
};

struct Config {
  int32_t priority;
  uint32_t run_timeout;
  uint32_t run_core_mask;
  uint8_t user_mem_weight;
  uint8_t user_mem_internal;
  uint8_t user_sram;
  uint8_t reserved[128];
};

struct Tensor {
  TensorMem *mem;
  TensorAttr *attr;
};

struct InitExtend {
  char *device_id;
  uint8_t reserved[128];
};

struct SdkVersion {
  char api_version[256];
  char drv_version[256];
};

struct CustomString {
  char string[1024];
};

} // namespace ztu::rk::rknpu3_abi
