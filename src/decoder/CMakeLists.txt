# Decoder module
add_library(decoder STATIC
    decoder.cpp
    decoder.hpp
)

# Include headers
target_include_directories(decoder PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

# Dependencies
target_link_libraries(decoder PUBLIC
    parser
)