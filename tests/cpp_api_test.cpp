#include <ztu/somemodelruntime_rknn.hpp>

#include <array>
#include <cassert>
#include <cstring>

int main() {
  namespace Ort = ztu::somemodelruntime::rknn;

  std::array<int64_t, 2> shape{2, 2};
  std::array<float, 4> input{1.0F, 2.0F, 3.0F, 4.0F};
  auto value = Ort::Value::CreateTensor<float>(
      Ort::MemoryInfo::CreateCpu(), input.data(), input.size(), shape.data(),
      shape.size());
  assert(value.IsTensor());
  assert(value.GetTensorTypeAndShapeInfo().GetElementType() ==
         Ort::TensorElementType::Float32);
  assert(value.GetTensorTypeAndShapeInfo().GetShape() ==
         std::vector<int64_t>(shape.begin(), shape.end()));
  assert(value.GetTensorData<float>() == input.data());

  std::array<float, 4> replacement{5.0F, 6.0F, 7.0F, 8.0F};
  value.Update(replacement.data(), sizeof(replacement),
               std::vector<int64_t>(shape.begin(), shape.end()),
               Ort::TensorElementType::Float32);
  assert(std::memcmp(input.data(), replacement.data(), sizeof(input)) == 0);

  Ort::AllocatorWithDefaultOptions allocator;
  auto owned = Ort::Value::CreateTensor<int32_t>(allocator, shape.data(),
                                                  shape.size());
  assert(owned.GetTensorSizeInBytes() == 4 * sizeof(int32_t));

  Ort::RunOptions run_options;
  assert(!run_options.GetDispatchBatch());
  run_options.SetDispatchBatch(true);
  assert(run_options.GetDispatchBatch());
  assert(std::string(Ort::MemoryInfo::CreateRknpu2().GetName()) == "rknpu2");
  return 0;
}
