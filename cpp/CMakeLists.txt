cmake_minimum_required(VERSION 3.20)
project(deep_learning_factor)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(./cmake_modules/FindSymForce.cmake)
set(TORCH_DIR /home/jagatpreet/software/libtorch)
set(GTSAM_DIR "/home/jagatpreet/software/gtsam/install_4_1")
find_package(GTSAM REQUIRED PATHS ${GTSAM_DIR})
find_package(GTest REQUIRED)
find_package(fmt REQUIRED)
find_package(Torch REQUIRED PATHS ${TORCH_DIR})

if(fmt_FOUND)
    message(STATUS "Format ${fmt_VERSION} Found")
endif()
if(Torch_FOUND)
    message(STATUS "Torch ${Torch_VERSION} Found")
endif()

# include generated code from symforce


# TARGETS

enable_testing()

add_executable(
    type_compatibility_test
    ./tests/type_compatibility_test.cpp
)
target_include_directories(type_compatibility_test
    PRIVATE
    ./include
    ${gtsam_INCLUDE_DIRS}
    ${SYMFORCE_INClUDE_DIR_SYM}
)
target_link_libraries(
    type_compatibility_test
    GTest::gtest_main
    gtsam
    ${SYMFORCE_LIBRARIES}
)
add_executable(
    regression
    ./src/regression.cpp
)
target_include_directories(
    regression
    PRIVATE
    ./include
    ${gtsam_INCLUDE_DIRS}
    ${SYMFORCE_INCLUDE_DIR_SYM}
)

target_link_libraries(
    regression
    GTest::gtest_main
    gtsam
    ${SYMFORCE_LIBRARIES}
)


add_library(
    fg_reader
    SHARED
    ./src/fg_reader.cpp
)
target_compile_options(
    fg_reader
    PRIVATE
    ${CMAKE_CXX_FLAGS}
    ${TORCH_CXX_FLAGS}
)
target_include_directories(
    fg_reader
    PRIVATE
    ./include
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(
    fg_reader
    ${TORCH_LIBRARIES}
)
set_property(TARGET fg_reader PROPERTY CXX_STANDARD 17)

add_library(
    fg_builder
    SHARED
    ./src/fg_builder.cpp
)
target_compile_options(
    fg_builder
    PRIVATE
    ${CMAKE_CXX_FLAGS}
    ${TORCH_CXX_FLAGS}
)
target_include_directories(
    fg_builder
    PRIVATE
    ./include
    ${SYMFORCE_INCLUDE_DIR_SYM}
    ${gtsam_INCLUDE_DIRS}
)
target_link_libraries(
    fg_builder
    gtsam
    ${SYMFORCE_LIBRARIES}
)
add_executable(fg_app
    ./src/fg_app.cpp
)
target_include_directories(fg_app
    PRIVATE
    ./include
    ${TORCH_INCLUDE_DIRS}
    ${gtsam_INCLUDE_DIRS}
)

target_link_libraries(fg_app ${fg_builder} ${fg_reader})

#include(GoogleTest)
#gtest_discover_tests(type_compatibility_test)
