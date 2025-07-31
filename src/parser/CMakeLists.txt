# Parser library
add_library(parser
    parser.cpp
    parser.hpp
)

# Include directories
target_include_directories(parser PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Link libraries
target_link_libraries(parser
    decoder
)

# Compiler-specific options
if(MSVC)
    target_compile_options(parser PRIVATE /W4)
else()
    target_compile_options(parser PRIVATE -Wall -Wextra -pedantic)
endif()