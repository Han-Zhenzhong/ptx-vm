# Logger module
add_library(logger STATIC
    logger.cpp
)

# Include headers
target_include_directories(logger PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

# No dependencies - this is a standalone utility module
