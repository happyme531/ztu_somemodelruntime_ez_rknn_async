# ztu_somemodelruntime_ez_rknn_async
### An ORT-style RKNPU2/RKNPU3 API for Python and C++

Supported Python versions: 3.7+

--------

## 🚀 Feature Comparison

| Feature | This Project | Official RKNPU2 Python runtime |
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

## C++ API

The public C++17 API is available from `<ztu/somemodelruntime_rknn.hpp>` in the `ztu::somemodelruntime::rknn` namespace. 

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

## RKNPU3 support

As current RKNPU3 Runtime SDK (v1.0.4) is very buggy, and not actually publicly released at all, current support is limited.

- Only zero-copy (`io_binding`) API is supported now.
- 4D input tensor shape info will be incorrect (different from the original model), because RKNPU3 runtime currently don't support querying the original model shapes, only the native ones. The runtime will still try to transpose a 4D input tensor into the native one without checking if the shape actually matches.
- Read model embed string don't work because it's currently not implemented in RKNPU3 runtime.

## Documentation

I don't know if it's a good idea to document this library, but anyway, there's an AI generated one that's generally okay: https://deepwiki.com/happyme531/ztu_somemodelruntime_ez_rknn_async , and another one https://mintlify.wiki/happyme531/ztu_somemodelruntime_ez_rknn_async
