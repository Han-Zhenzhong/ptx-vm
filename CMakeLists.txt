cmake_minimum_required(VERSION 3.14)
project(ptx_vm VERSION 1.0.0 LANGUAGES CXX)

# Project information
set(PROJECT_AUTHOR "Zhenzhong Han <zhenzhong.han@qq.com>")
set(PROJECT_DESCRIPTION "A virtual machine implementation for executing NVIDIA PTX (Parallel Thread Execution) intermediate code")

# Require C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler-specific options
if(MSVC)
    # Enable exceptions and RTTI for MSVC
    add_compile_options(/EHsc /GR)
else()
    # Enable exceptions and RTTI for GCC/Clang
    add_compile_options(-fexceptions -frtti)
endif()

# Options
option(BUILD_TESTS "Build test programs" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_DOCUMENTATION "Build documentation" ON)

add_compile_options(-g)

# Add subdirectories
add_subdirectory(src/logger)
add_subdirectory(src/core)
add_subdirectory(src/decoder)
add_subdirectory(src/execution)
add_subdirectory(src/memory)
add_subdirectory(src/optimizer)
add_subdirectory(src/debugger)
add_subdirectory(src/host)
add_subdirectory(src/parser)
add_subdirectory(src/registers)

# Main executable
add_executable(ptx_vm
    src/main.cpp
)

# Link libraries to main executable
target_link_libraries(ptx_vm
    logger
    core
    decoder
    execution
    memory
    optimizer
    debugger
    host
    parser
    registers
)

# Include directories
target_include_directories(ptx_vm PRIVATE
    include
    src
)

# Compiler-specific options for main executable
if(MSVC)
    target_compile_options(ptx_vm PRIVATE /W4)
else()
    target_compile_options(ptx_vm PRIVATE -Wall -Wextra -pedantic)
endif()

# Build tests if enabled
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Build examples if enabled
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
