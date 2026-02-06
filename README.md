# ztu_somemodelruntime_ez_rknn_async
### A better RKNPU2 python API

--------

## 🚀 Feature Comparison

| Feature | This Project | Official SDK |
| :--- | :---: | :---: |
| **Model Loading & Basic Inference** | ✅ Supported | ✅ Supported |
| **Multi-core Tensor Parallel Inference** | ❌ Not Planned | ✅ Supported |
| **Multi-core Data Parallel Inference** | ✅ Supported | ❌ Not Supported |
| **Pipeline-based Async Inference** | ✅ Supported | ⚠️ Limited (Depth = 1) |
| **True Async Inference (Callback/Future)** | ✅ Supported | ❌ Not Supported |
| **Multi-batch Data Parallel Inference** | ✅ Supported | ⚠️ Limited (Fixed batch/4D only) |
| **Custom Operator Plugins** | 🏗️ TODO | ❌ Not Supported |
| **API Style** | 🚀 ORT-like (Easy migration) | ⚙️ Proprietary (Complex) |
| **Zero Dependencies** | ✅ Yes (NumPy only) | ❌ No |
| **Break Other Packages** | ✅ No | ⚠️ Yes (https://github.com/airockchip/rknn-toolkit2/issues/414) |
| **Open Source** | 🔓 Yes (AGPLv3) | 🔒 No |
