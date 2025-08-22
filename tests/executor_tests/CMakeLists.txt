# Executor tests
add_executable(executor_tests
    basic_math_tests.cpp
    control_flow_tests.cpp
)

# Link with required libraries
target_link_libraries(executor_tests
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

# Add test to CTest
add_test(NAME executor_tests_run COMMAND executor_tests)