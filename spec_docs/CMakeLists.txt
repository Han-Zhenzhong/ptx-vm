cmake_minimum_required(VERSION 3.14)
project(ptx_vm_docs)

# Add documentation files
set(DOCS
    README.md
    docs/user_guide.md
    docs/developer_guide.md
    docs/api_documentation.md
    docs/reconvergence_mechanism.md
    docs/memory_optimizations.md
    docs/cuda_binary_loader.md
    docs/performance_testing.md
    docs/warp_scheduler.md
    docs/predicate_handler.md
    docs/divergence_handling.md
    docs/memory_optimization_examples.md
    docs/divergence_performance_testing.md
    docs/divergence_handling_examples.md
    docs/next_phase_development_plan.md
)

# Create documentation target
add_custom_target(docs
    COMMAND ${CMAKE_COMMAND} -E echo "Documentation is available in markdown format"
    COMMAND ${CMAKE_COMMAND} -E echo "Use markdown viewer or converter to view documentation"
)
