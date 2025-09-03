cmake_minimum_required(VERSION 3.14)
project(ptx_vm_tests)

# Include GoogleTest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Add test directories
# add_subdirectory(executor_tests)
# add_subdirectory(memory_tests)
# add_subdirectory(performance_benchmarks)

# Add system tests
add_executable(system_tests
    system_tests/smoke_test.cpp
    # system_tests/performance_test.cpp
    # system_tests/memory_performance_test.cpp
    # system_tests/reconvergence_performance_test.cpp
    # system_tests/executor_performance_test.cpp
    # system_tests/divergence_performance_test.cpp
)

target_compile_options(system_tests PRIVATE -g)

# Link system tests
target_link_libraries(system_tests
    PRIVATE
    core
    decoder
    execution
    memory
    optimizer
    debugger
    host
    parser
    registers
    ${GTEST_BOTH_LIBRARIES}
    pthread
)

# Add test targets
add_test(NAME system_tests_run COMMAND system_tests)

# Copy examples directory to the same location as the system_tests executable
add_custom_command(TARGET system_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/examples
    ${CMAKE_CURRENT_BINARY_DIR}/examples
    COMMENT "Copying examples directory to build directory"
)

# Add performance test target
# add_executable(performance_tests
#     system_tests/performance_test.cpp
#     system_tests/memory_performance_test.cpp
#     system_tests/reconvergence_performance_test.cpp
#     system_tests/executor_performance_test.cpp
#     system_tests/divergence_performance_test.cpp
# )
# 
# target_link_libraries(performance_tests
#     PRIVATE
#     core
#     decoder
#     execution
#     memory
#     optimizer
#     debugger
#     host
#     parser
#     registers
#     ${GTEST_BOTH_LIBRARIES}
#     pthread
# )
# 
# add_test(NAME performance_tests_run COMMAND performance_tests)