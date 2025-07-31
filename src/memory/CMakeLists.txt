# Memory subsystem module
add_library(memory STATIC
    ${PROJECT_SOURCE_DIR}/include/memory.hpp
    ${PROJECT_SOURCE_DIR}/src/memory/memory.cpp
    ${PROJECT_SOURCE_DIR}/src/memory/memory.hpp
    ${PROJECT_SOURCE_DIR}/src/memory/memory_optimizer.cpp
    ${PROJECT_SOURCE_DIR}/src/memory/memory_optimizer.hpp
)

# Include headers
target_include_directories(memory PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

# Dependencies
target_link_libraries(memory PRIVATE
    core
)