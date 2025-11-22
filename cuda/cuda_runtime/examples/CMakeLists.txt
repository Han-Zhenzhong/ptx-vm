# Example programs using libptxrt

cmake_minimum_required(VERSION 3.10)

project(ptxrt_examples LANGUAGES CXX)

# Find CUDA (optional, for NVCC compilation)
find_package(CUDA QUIET)

# Include ptxrt headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

# Link ptxrt library
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../build)

# If CUDA is found, we can compile .cu files with NVCC
if(CUDA_FOUND)
    message(STATUS "CUDA found, using NVCC for .cu files")
    
    cuda_add_executable(simple_add_nvcc simple_add.cu)
    target_link_libraries(simple_add_nvcc ptxrt)
    
else()
    message(STATUS "CUDA not found, examples can be built manually with Clang")
    message(STATUS "Example: clang++ simple_add.cu --cuda-path=/usr/local/cuda --cuda-gpu-arch=sm_61 -I.. -L../build -lptxrt -o simple_add")
endif()

# Note: To use Clang instead of NVCC, compile manually:
# clang++ simple_add.cu --cuda-path=/usr/local/cuda --cuda-gpu-arch=sm_61 -I.. -L../build -lptxrt -o simple_add
