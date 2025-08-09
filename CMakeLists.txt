cmake_minimum_required(VERSION 3.16)
project(ultra_fast_hawkes_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

# CUDA library
add_library(hawkes_gpu_kernels STATIC hawkes_gpu_kernels.cu)
set_property(TARGET hawkes_gpu_kernels PROPERTY CUDA_ARCHITECTURES 52)

# SIMD library (without problematic flags)
add_library(hawkes_simd_processors STATIC hawkes_simd_processors.cpp)
target_link_libraries(hawkes_simd_processors OpenMP::OpenMP_CXX)
target_compile_options(hawkes_simd_processors PRIVATE -mavx512f -mavx512dq -mavx512vl -Wno-psabi)

# Engine library
add_library(hawkes_engine STATIC hawkes_engine.cpp)
target_link_libraries(hawkes_engine hawkes_gpu_kernels hawkes_simd_processors)

# C API
add_library(hawkes_c_api STATIC hawkes_c_api.cpp)
target_link_libraries(hawkes_c_api hawkes_engine)

# Executables
add_executable(hawkes_example example_usage.cpp)
target_link_libraries(hawkes_example hawkes_engine)

add_executable(hawkes_benchmark benchmark.cpp)
target_link_libraries(hawkes_benchmark hawkes_engine)

add_executable(hawkes_test test_hawkes.cpp)
target_link_libraries(hawkes_test hawkes_engine)

# Python binding
#find_package(pybind11 QUIET)
#if(pybind11_FOUND)
#    pybind11_add_module(pyhawkes python_binding.cpp)
#    target_link_libraries(pyhawkes PRIVATE hawkes_engine)
#endif()
target_compile_options(hawkes_simd_processors PRIVATE -mavx512f -mavx512dq -mavx512vl -Wno-psabi)

# Add CUDA include directories to all targets
target_include_directories(hawkes_engine PRIVATE ${CUDA_INCLUDE_DIRS})
target_include_directories(hawkes_c_api PRIVATE ${CUDA_INCLUDE_DIRS})
target_include_directories(hawkes_example PRIVATE ${CUDA_INCLUDE_DIRS})
target_include_directories(hawkes_benchmark PRIVATE ${CUDA_INCLUDE_DIRS})
target_include_directories(hawkes_test PRIVATE ${CUDA_INCLUDE_DIRS})
