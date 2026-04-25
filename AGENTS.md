## 项目编译方法

python3 -m pip wheel . -w dist --no-deps --config-settings=cmake.build-type=Debug

注意：这个项目不提供c++ api，c++那边随便改，想怎么改都行不用管兼容性，python那边兼容就行
