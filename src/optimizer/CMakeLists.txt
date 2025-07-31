# Optimizer module
add_library(optimizer STATIC
    register_allocator.hpp
    register_allocator.cpp
    instruction_scheduler.hpp
    instruction_scheduler.cpp
)

# Include headers
target_include_directories(optimizer PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

# Dependencies
target_link_libraries(optimizer PRIVATE
    core
)