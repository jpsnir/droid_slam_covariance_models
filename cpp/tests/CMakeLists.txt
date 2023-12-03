add_executable(
    type_compatibility_test
    ./type_compatibility_test.cpp
)
target_include_directories(type_compatibility_test
    PRIVATE
    ${CMAKE_SOURCE_DIR}/droid_cvm/include
    ${gtsam_INCLUDE_DIRS}
    ${SYMFORCE_INClUDE_DIR_SYM}
)
target_link_libraries(
    type_compatibility_test
    GTest::gtest_main
    gtsam
    ${SYMFORCE_LIBRARIES}
)

add_executable(test_expressions ./test_expression_factor.cpp)
target_include_directories(test_expressions PRIVATE
    ${gtsam_INCLUDE_DIRS} ${SYMFORCE_INCLUDE_DIR_SYM} ${spdlog_INCLUDE_DIRS}
    ${fmt_INCLUDE_DIRS}
)
target_link_libraries(test_expressions GTest::gtest_main gtsam
    ${SYMFORCE_LIBRARIES} spdlog::spdlog
)
include(GoogleTest)
gtest_discover_tests(test_expressions)
add_test(NAME test_expressions COMMAND test_expressions)
