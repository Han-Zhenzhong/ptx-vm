# Examples CMakeLists.txt

# Create a custom target for examples
add_custom_target(examples
    COMMAND ${CMAKE_COMMAND} -E echo "PTX VM Examples:"
    COMMAND ${CMAKE_COMMAND} -E echo "  - simple_math_example.ptx"
    COMMAND ${CMAKE_COMMAND} -E echo "  - control_flow_example.ptx" 
    COMMAND ${CMAKE_COMMAND} -E echo "  - memory_ops_example.ptx"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Documentation:"
    COMMAND ${CMAKE_COMMAND} -E echo "  - memory_optimization_examples.md"
    COMMAND ${CMAKE_COMMAND} -E echo "  - divergence_handling_examples.md"
)

# Install example files
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION examples
    FILES_MATCHING 
    PATTERN "*.ptx"
    PATTERN "*.md"
)