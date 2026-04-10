# ztu_somemodelruntime_ez_rknn_async
### A better RKNPU2 python API

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
| **Custom Operator Plugins** | ✅ Supported | ❌ Not Supported |
| **Read model embed string** | ✅ Supported | ❌ Not Supported |
| **API Style** | 🚀 ORT-like (Easy migration) | ⚙️ Proprietary (Complex) |
| **Zero Dependencies** | ✅ Yes (NumPy only) | ❌ No |
| **Break Other Packages** | ✅ No | ⚠️ Yes (https://github.com/airockchip/rknn-toolkit2/issues/414) |
| **Open Source** | 🔓 Yes (AGPLv3) | 🔒 No |

## Installation

```bash
pip install ztu-somemodelruntime-ez-rknn-async
```

or manually build (`python -m build`) and install by yourself.

## Usage

The usage is similar to ONNXRuntime Python API, you can load a `.rknn` model and use `run()` or `run_async()` to do inference. To use these advanced features, you need to configure corresponding `provider_options` or `run_options` (refer to the documentation).

## Documentation 

I don't know if it's a good idea to document this library, but anyway, there's an AI generated one that's generally okay: https://deepwiki.com/happyme531/ztu_somemodelruntime_ez_rknn_async , and another one https://mintlify.wiki/happyme531/ztu_somemodelruntime_ez_rknn_async

