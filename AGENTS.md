## 项目编译方法

python3 -m pip wheel . -w dist --no-deps --config-settings=cmake.build-type=Debug

项目同时提供 Python API 和 `<ztu/somemodelruntime_rknn.hpp>` C++17 API。Python API 必须保持兼容；
C++ API 在 1.0 前尽量保持源码兼容，但不承诺二进制 ABI 兼容。

C++ 库构建与测试：

cmake -S . -B build/cpp -DZTU_SOMEMODELRUNTIME_RKNN_BUILD_PYTHON=OFF
cmake --build build/cpp -j
ctest --test-dir build/cpp --output-on-failure
