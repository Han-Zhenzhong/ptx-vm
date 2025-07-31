# Host API module
add_library(host STATIC
    host_api.hpp
    host_api.cpp
    cli_interface.hpp
    cli_interface.cpp
    cuda_binary_loader.hpp
    cuda_binary_loader.cpp
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
    core
    optimizer
    # 添加对host模块自身的引用
    host
)