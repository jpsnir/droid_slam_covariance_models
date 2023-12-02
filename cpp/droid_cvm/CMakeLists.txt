add_executable(
    regression
    ./src/apps/regression.cpp
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
    ./src/libs/fg_reader.cpp
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
    ./src/libs/fg_builder.cpp
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
#add_executable(fg_app
#    ./src/include/fg_app.cpp
#)
#target_include_directories(fg_app
#    PRIVATE
#    ./include
#    ${TORCH_INCLUDE_DIRS}
#    ${gtsam_INCLUDE_DIRS}
#)
#target_link_libraries(fg_app ${fg_builder} ${fg_reader})
#
