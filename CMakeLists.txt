cmake_minimum_required(VERSION 3.16)
project(ultra_fast_hawkes_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
include(CheckCXXCompilerFlag)

# Test for SIMD support
check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)

# CUDA library
add_library(hawkes_gpu_kernels STATIC hawkes_gpu_kernels.cu)
set_property(TARGET hawkes_gpu_kernels PROPERTY CUDA_ARCHITECTURES 75)

# SIMD library with automatic detection
add_library(hawkes_simd_processors STATIC hawkes_simd_processors.cpp)
target_link_libraries(hawkes_simd_processors OpenMP::OpenMP_CXX)

if(COMPILER_SUPPORTS_AVX512)
    message(STATUS "✅ Using AVX-512 (16-float SIMD)")
    target_compile_options(hawkes_simd_processors PRIVATE -mavx512f -mavx512dq -mavx512vl)
    target_compile_definitions(hawkes_simd_processors PRIVATE USE_AVX512=1)
elseif(COMPILER_SUPPORTS_AVX2)
    message(STATUS "✅ Using AVX-256 (8-float SIMD)")
    target_compile_options(hawkes_simd_processors PRIVATE -mavx2 -mfma)
    target_compile_definitions(hawkes_simd_processors PRIVATE USE_AVX256=1)
else()
    message(STATUS "⚠️  Using scalar operations")
    target_compile_definitions(hawkes_simd_processors PRIVATE USE_SCALAR=1)
endif()

# Engine library
add_library(hawkes_engine STATIC hawkes_engine.cpp)
target_link_libraries(hawkes_engine hawkes_gpu_kernels hawkes_simd_processors)
target_include_directories(hawkes_engine PRIVATE ${CUDA_INCLUDE_DIRS})

# C API
add_library(hawkes_c_api STATIC hawkes_c_api.cpp)
target_link_libraries(hawkes_c_api hawkes_engine)
target_include_directories(hawkes_c_api PRIVATE ${CUDA_INCLUDE_DIRS})

# Executables
add_executable(hawkes_example example_usage.cpp)
target_link_libraries(hawkes_example hawkes_engine)
target_include_directories(hawkes_example PRIVATE ${CUDA_INCLUDE_DIRS})

add_executable(hawkes_benchmark benchmark.cpp)
target_link_libraries(hawkes_benchmark hawkes_engine)
target_include_directories(hawkes_benchmark PRIVATE ${CUDA_INCLUDE_DIRS})

add_executable(hawkes_test test_hawkes.cpp)
target_link_libraries(hawkes_test hawkes_engine)
target_include_directories(hawkes_test PRIVATE ${CUDA_INCLUDE_DIRS})
