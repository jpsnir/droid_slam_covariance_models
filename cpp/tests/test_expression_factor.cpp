/*
 * that can be done with expression factors.
 * The main goals of these sense tests are to understand the usage of expression
 * factors in gtsam library and how auto differentiation is done in gtsam.
 * In addition, we will understand try to compare the output of expression
 * factors generated from symforce and gtsam expression factors.
 *
 * The basic idea is the expression type reuses the jacobians  defined for
 * the elementary operations defined in gtsam for different types.
 * And uses chain rule to give the new gradients of expressions.
 *
 * The set of tests are also a tutorial to understand usage of expression and
 */
#include <gtsam/base/Vector.h>
#include <iostream>
#include <gtest/gtest.h>
#include <gtsam/nonlinear/Expression.h>
#include <gtsam/nonlinear/ExpressionFactor.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/Values.h>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/fmt.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <string>
#include <vector>
using namespace gtsam;

TEST(ExpressionUsage, ConstructionTest){
    // construct constants and retreving their value
    auto a = Point2(3,4);
    Expression<Point2> k_expr(Point2(3,4));
    Values v; std::vector<Matrix> H;
    Point2 e = k_expr.value(v, H);
    fmt::print(" values dimension : {} \n", v.dim());
    ASSERT_EQ(a, e);

    // value of constant
    v.insert(1, gtsam::Vector1(1));
    e = k_expr.value(v, H);
    //ASSERT_EQ(a, e);

}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
