# CUDA Runtime API Implementation (libptxrt)
# This library provides a drop-in replacement for NVIDIA's libcudart

cmake_minimum_required(VERSION 3.10)

project(ptxrt VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
)
set_target_properties(ptxrt_static PROPERTIES
    OUTPUT_NAME ptxrt
    POSITION_INDEPENDENT_CODE ON
)

# Create shared library
add_library(ptxrt_shared SHARED ${PTXRT_SOURCES})
target_include_directories(ptxrt_shared PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(ptxrt_shared PROPERTIES
    OUTPUT_NAME ptxrt
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
)

# Installation rules
install(TARGETS ptxrt_static ptxrt_shared
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${PTXRT_HEADERS}
    DESTINATION include/ptxrt
)

# Create a CMake config file for easy integration
include(CMakePackageConfigHelper)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/ptxrtConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ptxrtConfig.cmake
    INSTALL_DESTINATION lib/cmake/ptxrt
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/ptxrtConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/ptxrtConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/ptxrtConfigVersion.cmake
    DESTINATION lib/cmake/ptxrt
)

# Print build information
message(STATUS "PTX Runtime Library Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Install Prefix: ${CMAKE_INSTALL_PREFIX}")
