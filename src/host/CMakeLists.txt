# Host API module
add_library(host STATIC
    host_api.cpp
    cli_interface.cpp
)

# Include headers
target_include_directories(host PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

# Add new source files for host module
set(SRCS
    ${SRCS}
    # Source files already added in add_library
)

# Dependencies
target_link_libraries(host PRIVATE
    logger
    core
    optimizer
)