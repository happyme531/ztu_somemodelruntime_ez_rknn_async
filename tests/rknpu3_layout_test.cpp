#include "rknpu3_layout.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>

int main() {
  namespace abi = ztu::rk::rknpu3_abi;

  abi::TensorAttr attr = {};
  attr.n_dims = 5;
  attr.shape[0] = 1;
  attr.shape[1] = 1;
  attr.shape[2] = 2;
  attr.shape[3] = 2;
  attr.shape[4] = 4;
  attr.n_stride = 5;
  attr.stride[0] = 16;
  attr.stride[1] = 16;
  attr.stride[2] = 8;
  attr.stride[3] = 4;
  attr.stride[4] = 1;
  attr.layout = abi::TensorLayout::Nc1hwc2;

  const std::array<uint16_t, 12> source = {0,  1,  2,  3,  10, 11,
                                           12, 13, 20, 21, 22, 23};
  std::array<uint16_t, 32> destination;
  destination.fill(0xffff);
  ztu::rk::copy_nchw_to_rknpu3_nc1hwc2(
      source.data(), sizeof(source), {1, 3, 2, 2}, destination.data(),
      sizeof(destination), sizeof(uint16_t), attr);

  const std::array<uint16_t, 16> expected = {0, 10, 20, 0, 1, 11, 21, 0,
                                             2, 12, 22, 0, 3, 13, 23, 0};
  for (size_t i = 0; i < expected.size(); ++i)
    assert(destination[i] == expected[i]);
  for (size_t i = expected.size(); i < destination.size(); ++i)
    assert(destination[i] == 0);

  bool rejected = false;
  try {
    ztu::rk::copy_nchw_to_rknpu3_nc1hwc2(
        source.data(), sizeof(source), {1, 5, 2, 2}, destination.data(),
        sizeof(destination), sizeof(uint16_t), attr);
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  assert(rejected);
  return 0;
}
