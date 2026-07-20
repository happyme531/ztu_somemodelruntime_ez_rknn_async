cmake_minimum_required(VERSION 3.18)
project(ztu_somemodelruntime_ez_rknn_async VERSION 0.10.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(ZTU_SOMEMODELRUNTIME_RKNN_BUILD_PYTHON "Build the Python extension" OFF)
option(ZTU_SOMEMODELRUNTIME_RKNN_INSTALL_DEVEL "Install C++ headers and CMake package" ON)

if(CMAKE_BUILD_TYPE MATCHES "Release")
    add_compile_options(-g1 -O3 -fno-omit-frame-pointer)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -rdynamic")
endif()

find_package(Threads REQUIRED)

add_library(ztu_somemodelruntime_rknn SHARED
    src/rknpu2_runtime_loader.cpp
    src/rknpu3_backend.cpp
    src/somemodelruntime_rknn.cpp)
add_library(ztu::somemodelruntime_rknn ALIAS ztu_somemodelruntime_rknn)
set_target_properties(ztu_somemodelruntime_rknn PROPERTIES
    EXPORT_NAME somemodelruntime_rknn
    OUTPUT_NAME ztu_somemodelruntime_rknn
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
if(ZTU_SOMEMODELRUNTIME_RKNN_INSTALL_DEVEL)
    set_target_properties(ztu_somemodelruntime_rknn PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION 0)
endif()
target_compile_definitions(ztu_somemodelruntime_rknn PRIVATE ZTU_SOMEMODELRUNTIME_RKNN_BUILDING_LIBRARY=1)
target_include_directories(ztu_somemodelruntime_rknn
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(ztu_somemodelruntime_rknn PRIVATE Threads::Threads dl)

if(BUILD_TESTING)
    add_executable(ztu_somemodelruntime_rknn_cpp_api_test tests/cpp_api_test.cpp)
    target_link_libraries(ztu_somemodelruntime_rknn_cpp_api_test PRIVATE ztu::somemodelruntime_rknn)
    add_test(NAME ztu_somemodelruntime_rknn_cpp_api_test COMMAND ztu_somemodelruntime_rknn_cpp_api_test)
endif()

if(ZTU_SOMEMODELRUNTIME_RKNN_BUILD_PYTHON)
    set(PYBIND11_STRIP OFF CACHE BOOL "Keep symbols in release for debugging" FORCE)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(_core NO_EXTRAS src/pybind_module.cpp)
    target_include_directories(_core PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(_core PRIVATE ztu_somemodelruntime_rknn)
    target_compile_definitions(_core PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES=1)
    set_target_properties(_core PROPERTIES INSTALL_RPATH "$ORIGIN")

    install(TARGETS _core ztu_somemodelruntime_rknn
        LIBRARY DESTINATION ztu_somemodelruntime_ez_rknn_async)
    install(FILES python/ztu_somemodelruntime_ez_rknn_async/__init__.py
        DESTINATION ztu_somemodelruntime_ez_rknn_async)
    install(FILES
      python/ztu_somemodelruntime_ez_rknn_async/options.py
      python/ztu_somemodelruntime_ez_rknn_async/__init__.pyi
      python/ztu_somemodelruntime_ez_rknn_async/py.typed
      DESTINATION ztu_somemodelruntime_ez_rknn_async)
endif()

if(ZTU_SOMEMODELRUNTIME_RKNN_INSTALL_DEVEL)
    install(TARGETS ztu_somemodelruntime_rknn
        EXPORT ztu_somemodelruntime_rknnTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(EXPORT ztu_somemodelruntime_rknnTargets
        FILE ztu_somemodelruntime_rknnTargets.cmake
        NAMESPACE ztu::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ztu_somemodelruntime_rknn)
    configure_package_config_file(
        cmake/ztu_somemodelruntime_rknnConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/ztu_somemodelruntime_rknnConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ztu_somemodelruntime_rknn)
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/ztu_somemodelruntime_rknnConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion)
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/ztu_somemodelruntime_rknnConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/ztu_somemodelruntime_rknnConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ztu_somemodelruntime_rknn)
endif()
