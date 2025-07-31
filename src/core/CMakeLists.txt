# Core VM module
add_library(core STATIC
    vm.hpp
    vm.cpp
    performance_counters.hpp
    performance_counters.cpp
)

# Include headers
target_include_directories(core PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

# Dependencies
target_link_libraries(core PRIVATE
    execution
    debugger
    optimizer
    host
)