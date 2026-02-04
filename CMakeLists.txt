cmake_minimum_required(VERSION 3.18)
project(ztu_somemodelruntime_ez_rknn_async LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(Threads REQUIRED)

add_library(rknnrt SHARED IMPORTED)
set_target_properties(rknnrt PROPERTIES
  IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/lib/librknnrt.so"
)

pybind11_add_module(_core src/pybind_module.cpp)

target_include_directories(_core PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(_core PRIVATE rknnrt Threads::Threads dl)

target_compile_definitions(_core PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES=1)

set_target_properties(_core PROPERTIES
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN"
)

install(TARGETS _core LIBRARY DESTINATION ztu_somemodelruntime_ez_rknn_async)
install(FILES lib/librknnrt.so DESTINATION ztu_somemodelruntime_ez_rknn_async)
install(FILES python/ztu_somemodelruntime_ez_rknn_async/__init__.py DESTINATION ztu_somemodelruntime_ez_rknn_async)
