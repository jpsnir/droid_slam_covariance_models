cmake_minimum_required(VERSION 3.20)
project(deep_learning_factor)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(./cmake_modules/FindSymForce.cmake)

set(GTSAM_DIR "/home/jagatpreet/software/gtsam/install_4_1")
find_package(GTSAM REQUIRED PATHS ${GTSAM_DIR})
find_package(fmt REQUIRED)
if(fmt_FOUND)
    message(STATUS "Format library found ${fmt_INCLUDE_DIRS}")
endif()
# include generated code from symforce
include_directories(
    ./src/gen
    ${gtsam_INCLUDE_DIRS}
)
add_executable(linear_regression ./src/linear_regression.cpp)
target_link_libraries(linear_regression gtsam fmt::fmt)
