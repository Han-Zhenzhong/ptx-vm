# CUDA Runtime API Implementation (libptxrt)
# This library provides a drop-in replacement for NVIDIA's libcudart

cmake_minimum_required(VERSION 3.10)

project(ptxrt VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find PTX VM include and library paths
set(PTXVM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
set(PTXVM_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
set(PTXVM_BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../build)

# Include directories for PTX VM
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PTXVM_INCLUDE_DIR}
    ${PTXVM_SRC_DIR}
)

# Link directories for PTX VM
link_directories(
    ${PTXVM_BUILD_DIR}
    ${PTXVM_BUILD_DIR}/src/host
    ${PTXVM_BUILD_DIR}/src/core
    ${PTXVM_BUILD_DIR}/src/logger
    ${PTXVM_BUILD_DIR}/src/decoder
    ${PTXVM_BUILD_DIR}/src/execution
    ${PTXVM_BUILD_DIR}/src/memory
    ${PTXVM_BUILD_DIR}/src/optimizer
    ${PTXVM_BUILD_DIR}/src/debugger
    ${PTXVM_BUILD_DIR}/src/parser
    ${PTXVM_BUILD_DIR}/src/registers
)

# Source files
set(PTXRT_SOURCES
    cuda_runtime.cpp
)

# Header files
set(PTXRT_HEADERS
    cuda_runtime.h
)

# Create static library
add_library(ptxrt_static STATIC ${PTXRT_SOURCES})
target_include_directories(ptxrt_static PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PTXVM_INCLUDE_DIR}
)
target_link_libraries(ptxrt_static PUBLIC
    host
    core
    logger
    decoder
    execution
    memory
    optimizer
    debugger
    parser
    registers
)
set_target_properties(ptxrt_static PROPERTIES
    OUTPUT_NAME ptxrt
    POSITION_INDEPENDENT_CODE ON
)

# Note: Shared library disabled because PTX VM libraries are not compiled with -fPIC
# To enable shared library, PTX VM must be rebuilt with CMAKE_POSITION_INDEPENDENT_CODE=ON

# Create shared library
# add_library(ptxrt_shared SHARED ${PTXRT_SOURCES})
# target_include_directories(ptxrt_shared PUBLIC
#     ${CMAKE_CURRENT_SOURCE_DIR}
#     ${PTXVM_INCLUDE_DIR}
# )
# target_link_libraries(ptxrt_shared PUBLIC
#     host
#     core
#     logger
#     decoder
#     execution
#     memory
#     optimizer
#     debugger
#     parser
#     registers
# )
# set_target_properties(ptxrt_shared PROPERTIES
#     OUTPUT_NAME ptxrt
#     VERSION ${PROJECT_VERSION}
#     SOVERSION 1
# )

# Installation rules
install(TARGETS ptxrt_static
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${PTXRT_HEADERS}
    DESTINATION include/ptxrt
)

# Print build information
message(STATUS "PTX Runtime Library Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Install Prefix: ${CMAKE_INSTALL_PREFIX}")
