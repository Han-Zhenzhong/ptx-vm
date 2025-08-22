# Memory tests
add_executable(memory_tests
    advanced_memory_tests.cpp
)

# Link with required libraries
target_link_libraries(memory_tests
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
add_test(NAME memory_tests_run COMMAND memory_tests)