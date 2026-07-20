# ztu_somemodelruntime_ez_rknn_async
### An ORT-style RKNPU2/RKNPU3 API for Python and C++

Supported Python versions: 3.7+

--------

## 🚀 Feature Comparison

| Feature | This Project | Official SDK |
| :--- | :---: | :---: |
| **Model Loading & Basic Inference** | ✅ Supported | ✅ Supported |
| **Multi-core Tensor Parallel Inference** | ✅ Supported | ✅ Supported |
| **Multi-core Data Parallel Inference** | ✅ Supported | ❌ Not Supported |
| **Pipeline-based Async Inference** | ✅ Supported | ⚠️ Limited (Depth = 1) |
| **True Async Inference (Callback/Future)** | ✅ Supported | ❌ Not Supported |
| **Multi-batch Data Parallel Inference** | ✅ Supported | ⚠️ Limited (Fixed batch/4D only) |
| **Zero-copy Inference** | ✅ Supported (via OrtValue/io_binding API) | ❌ Not Supported |
| **Multi model weight sharing** | ✅ Supported | ❌ Not Supported |
| **Custom Operator Plugins** | ✅ Supported | ❌ Not Supported |
| **Read model embed string** | ✅ Supported | ❌ Not Supported |
| **Python and C++ APIs** | ✅ Supported | ⚠️ Proprietary C API |
| **API Style** | 🚀 ORT-like (Easy migration) | ⚙️ Proprietary (Complex) |
| **Zero Dependencies** | ✅ Yes (NumPy only) | ❌ No |
| **Break Other Packages** | ✅ No | ⚠️ Yes (https://github.com/airockchip/rknn-toolkit2/issues/414) |
| **Open Source** | 🔓 Yes (AGPLv3) | 🔒 No |

## Installation

```bash
pip install ztu-somemodelruntime-ez-rknn-async
```

or manually build a wheel:

```bash
python3 -m pip wheel . -w dist --no-deps
```

## Usage

The usage is similar to ONNXRuntime Python API, you can load a `.rknn` model and use `run()` or `run_async()` to do inference. To use these advanced features, you need to configure corresponding `provider_options` or `run_options` (refer to the documentation).

### Execution providers

The model header selects the provider automatically: legacy RKNPU2 models use
the `RKNN` magic and RKNPU3 models use `RKNX`. They can also be selected
explicitly as `RKNPU2ExecutionProvider` and `RKNPU3ExecutionProvider`.
RKNPU3 path-based models require the companion `.weight` file with the same
basename. For model bytes, pass `weight_path` in the RKNPU3 provider options.

Runtime libraries are loaded dynamically. The default lookup uses bare library
names and therefore follows the platform dynamic-linker configuration and
`LD_LIBRARY_PATH`; the project does not contain built-in SDK paths. For example:

```bash
export LD_LIBRARY_PATH=/path/to/rknn3-api/Linux/aarch64:$LD_LIBRARY_PATH
```

Release wheels and C++ packages are built for Linux x86_64 and aarch64. RKNN
runtime libraries are not linked or bundled; install the matching vendor
runtime separately on the machine that performs inference.

The initial RKNPU3 implementation intentionally exposes only synchronous,
single-context native zero-copy inference through `IoBinding`:

```python
import numpy as np
import ztu_somemodelruntime_ez_rknn_async as ezrknn

session = ezrknn.InferenceSession("model.rknn")
assert session.get_providers() == ["RKNPU3ExecutionProvider"]

input_arg = session.get_native_inputs()[0]
output_arg = session.get_native_outputs()[0]
input_value = ezrknn.OrtValue.ortvalue_from_numpy(
    np.zeros(input_arg.shape, dtype=np.float16),
    "rknpu3",
    session=session,
    name=input_arg.name,
    io_kind="input",
)

binding = session.io_binding()
binding.bind_ortvalue_input(input_arg.name, input_value)
binding.bind_output(output_arg.name, "rknpu3")
session.run_with_iobinding(binding)
output = binding.get_outputs()[0].numpy()
```

RKNPU3 tensor shapes in this API are native runtime shapes. In particular, a
4D ONNX NCHW input may become a 5D `NC1HWC2` tensor with padded channels. Use
the reported `layout`, `shape`, `strides`, and `aligned_size`; do not assume the
ONNX shape.

As a temporary compatibility path, copied CPU inputs supplied through
`bind_cpu_input()` or `OrtValue.ortvalue_from_numpy()` may use 4D NCHW shape
when the native RKNPU3 input is 5D `NC1HWC2`; the provider packs and
zero-fills the native buffer. Native OrtValues and dma-bufs still require the
reported native shape. Exact logical-shape metadata remains unavailable until
the runtime implements `RKNN3_QUERY_ORIGIN_INPUT_ATTR` correctly.

## C++ API

The public C++17 API is available from `<ztu/somemodelruntime_rknn.hpp>` in the
`ztu::somemodelruntime::rknn` namespace. It intentionally implements the dense-tensor subset
needed by RKNN instead of defining or depending on the real `Ort` namespace.
Code that does not include ONNX Runtime in the same translation unit can use a
namespace alias to minimize migration changes:

```cpp
#include <ztu/somemodelruntime_rknn.hpp>

#include <array>

namespace Ort = ztu::somemodelruntime::rknn;

int main() {
  Ort::Env env;
  Ort::RknnProviderOptions provider_options;
  provider_options.layout = Ort::Layout::Nchw;

  Ort::SessionOptions session_options;
  session_options.SetRknnProviderOptions(provider_options);
  Ort::Session session(env, "model.rknn", session_options);

  std::array<int64_t, 4> shape{1, 3, 224, 224};
  std::array<float, 1 * 3 * 224 * 224> input{};
  auto input_value = Ort::Value::CreateTensor<float>(
      Ort::MemoryInfo::CreateCpu(), input.data(), input.size(),
      shape.data(), shape.size());

  const char *input_names[] = {session.GetInputName(0).c_str()};
  auto outputs = session.Run(Ort::RunOptions{}, input_names, &input_value, 1,
                             nullptr, 0);
  const float *output = outputs[0].GetTensorData<float>();
  (void)output;
}
```

For zero-copy inference, construct an `Ort::IoBinding`, bind CPU or
provider-matching `Value` objects, and call `session.Run(run_options, binding)`.
Session-aware RKNPU2/RKNPU3 buffers and external dma-bufs are created with
`CreateIoValue()` and `CreateIoValueFromDmaBuf()`. Buffer identities remain
explicit (`rknpu2` versus `rknpu3`) because their runtime objects are different.

Build and install the native package with:

```bash
cmake -S . -B build/cpp -DZTU_SOMEMODELRUNTIME_RKNN_BUILD_PYTHON=OFF
cmake --build build/cpp -j
cmake --install build/cpp --prefix /your/prefix
```

Downstream projects can then use:

```cmake
find_package(ztu_somemodelruntime_rknn CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE ztu::somemodelruntime_rknn)
```

The RKNPU2 regular `Session::Run` path preserves the Python API's copied
float-output behavior. Native dtype/layout buffers and true zero-copy operation
are exposed through `IoBinding`. Sparse tensors, string tensors, maps/sequences,
and graph optimization settings are outside this RKNN-focused compatibility
subset.

## Documentation

I don't know if it's a good idea to document this library, but anyway, there's an AI generated one that's generally okay: https://deepwiki.com/happyme531/ztu_somemodelruntime_ez_rknn_async , and another one https://mintlify.wiki/happyme531/ztu_somemodelruntime_ez_rknn_async
