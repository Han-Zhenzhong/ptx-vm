# Examples CMakeLists.txt

# Create executable for parameter passing example
add_executable(parameter_passing_example
    parameter_passing_example.cpp
)

# Create executable for execution result demo
add_executable(execution_result_demo
    execution_result_demo.cpp
)

# Link with host library
target_link_libraries(parameter_passing_example
    host
    decoder
    parser
)

target_link_libraries(execution_result_demo
    host
    decoder
    parser
)

# Include directories
target_include_directories(parameter_passing_example PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_include_directories(execution_result_demo PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Compiler-specific options
if(MSVC)
    target_compile_options(parameter_passing_example PRIVATE /W4)
    target_compile_options(execution_result_demo PRIVATE /W4)
else()
    target_compile_options(parameter_passing_example PRIVATE -Wall -Wextra -pedantic)
    target_compile_options(execution_result_demo PRIVATE -Wall -Wextra -pedantic)
endif()

# Create a custom target for examples
add_custom_target(examples
    COMMAND ${CMAKE_COMMAND} -E echo "PTX VM Examples:"
    COMMAND ${CMAKE_COMMAND} -E echo "  - simple_math_example.ptx"
    COMMAND ${CMAKE_COMMAND} -E echo "  - control_flow_example.ptx" 
    COMMAND ${CMAKE_COMMAND} -E echo "  - memory_ops_example.ptx"
    COMMAND ${CMAKE_COMMAND} -E echo "  - parameter_passing_example.ptx"
    COMMAND ${CMAKE_COMMAND} -E echo "  - parameter_passing_example (executable)"
    COMMAND ${CMAKE_COMMAND} -E echo "  - execution_result_demo (executable)"
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

# Install the example executables
install(TARGETS parameter_passing_example execution_result_demo
    DESTINATION examples
)