# Debugger module
add_library(debugger STATIC
    debugger.hpp
    debugger.cpp
)

# Include headers
target_include_directories(debugger PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

# Dependencies
target_link_libraries(debugger PRIVATE
    core
    execution
)