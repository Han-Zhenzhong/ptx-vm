# Performance benchmarks
add_executable(register_allocation_benchmark 
    register_allocation_benchmark.cpp
)

# Link with required libraries
target_link_libraries(register_allocation_benchmark PRIVATE
    core
    optimizer
    execution
    gtest_main
    benchmark::benchmark_main
)

# Add to test suite
add_test(NAME RegisterAllocationBenchmark COMMAND register_allocation_benchmark)