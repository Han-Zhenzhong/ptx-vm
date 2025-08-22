# Create an interface library for the header files
add_library(include INTERFACE)

# Specify the include directories for this interface library
target_include_directories(include INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)