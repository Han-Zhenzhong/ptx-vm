# Registers library
add_library(registers
    register_bank.cpp
    register_bank.hpp
)

# Include directories
target_include_directories(registers PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Compiler-specific options
if(MSVC)
    target_compile_options(registers PRIVATE /W4)
else()
    target_compile_options(registers PRIVATE -Wall -Wextra -pedantic)
endif()