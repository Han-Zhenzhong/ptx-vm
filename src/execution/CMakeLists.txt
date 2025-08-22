# Execution engine module
add_library(execution STATIC
    ${PROJECT_SOURCE_DIR}/src/execution/executor.cpp
    ${PROJECT_SOURCE_DIR}/src/execution/control_flow_graph.cpp
    ${PROJECT_SOURCE_DIR}/src/execution/divergence_reconvergence.cpp
    ${PROJECT_SOURCE_DIR}/src/execution/predicate_handler.cpp
    ${PROJECT_SOURCE_DIR}/src/execution/reconvergence_mechanism.cpp
    ${PROJECT_SOURCE_DIR}/src/execution/warp_scheduler.cpp
)

# Include headers
target_include_directories(execution PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

# Add new source files for execution module
set(SRCS
    ${SRCS}
    control_flow_graph.cpp
    divergence_reconvergence.cpp
    predicate_handler.cpp
    reconvergence_mechanism.cpp
    warp_scheduler.cpp
)

# Dependencies
target_link_libraries(execution PRIVATE
    core
    memory
    optimizer
    registers
)