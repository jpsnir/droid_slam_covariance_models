cmake_minimum_required(VERSION 3.20)
project(deep_learning_factor)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(./cmake_modules/FindSymForce.cmake)

set(GTSAM_DIR "/home/jagatpreet/software/gtsam/install_4_1")
find_package(GTSAM REQUIRED PATHS ${GTSAM_DIR})
find_package(GTest REQUIRED)
find_package(fmt REQUIRED)
if(fmt_FOUND)
    message(STATUS "Format ${fmt_VERSION} Found")
endif()
# include generated code from symforce
include_directories(
    ./include
    ${gtsam_INCLUDE_DIRS}
)
include_directories(${SYMFORCE_INCLUDE_DIR_SYM})
enable_testing()
add_executable(
    type_compatibility_test
    ./tests/type_compatibility_test.cpp
)
add_executable(
    regression
    ./src/regression.cpp
)

target_link_libraries(
    regression
    GTest::gtest_main
    gtsam
    ${SYMFORCE_LIBRARIES}
)

target_link_libraries(
    type_compatibility_test
    GTest::gtest_main
    gtsam
    fmt::fmt
    ${SYMFORCE_LIBRARIES}
)

include(GoogleTest)
gtest_discover_tests(type_compatibility_test)
