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
#include <cstdio>
#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Expression.h>
#include <gtsam/nonlinear/expressions.h>
#include <gtsam/nonlinear/ExpressionFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <iostream>
#include <spdlog/fmt/fmt.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>
using namespace gtsam;

TEST(ExpressionUsage, ConstantTest) {
  // construct constants and retreving their value
  auto a = Point2(3, 4);
  Expression<Point2> k_expr(Point2(3, 4));
  Values v;
  std::vector<Matrix> H;
  Point2 e = k_expr.value(v, H);
  fmt::print(" values dimension : {} \n", v.dim());
  ASSERT_EQ(a, e);

  // value of constant
  v.insert(1, gtsam::Vector1(1));
  e = k_expr.value(v, H);
  ASSERT_EQ(a, e);
  fmt::print("size of jacobian list :{}", H.size());
  ASSERT_EQ(H.size(), 0);
}

TEST(ExpressionUsage, VariableTest) {
  // expression variable
  Key k = Symbol('x', 1);
  Point2 pt(2, 3);
  Expression<Point2> k_expr(k);
  Values v;
  v.insert(k, pt);

  // size of the vector is equal to the number of keys
  // otherwise we get a runtime error while evaluating
  // the expression.
  std::vector<Matrix> H(1);
  Point2 e = k_expr.value(v, H);
  std::cout << "Actual point = \n"
            << pt << "\n, Expected point = \n"
            << e << std::endl;
  ASSERT_EQ(pt, e);
  // Eigen matrix is empty, so size = 0
  ASSERT_TRUE(H[1].size() == 0);
}

TEST(ExpressionUsage, AdditionTest) {
  Key k1 = Symbol('x', 1);
  Key k2 = Symbol('x', 2);
  Point2 pt1(10, 20);
  Point2 pt2(2, 3);
  // Adding same keys
  Expression<Point2> expr1(k1);
  Expression<Point2> expr2(k1);
  Expression<Point2> expr3(k2);
  Expression<Point2> expr_sum_1 = expr1 + expr2;
  Expression<Point2> expr_sum_2 = expr1 + expr2 + expr3;
  Values v;
  v.insert(k1, pt1);
  Point2 e = expr_sum_1.value(v);
  std::cout << "Expected value:\n " << e << std::endl;
  // In eigen matrices, 2*pt1 is ambiguous (may be pointer indirection), use
  // pt1*2 which is ok.
  ASSERT_EQ(pt1 + pt1, e);

  // This throws an error because we have not provided
  // values to all the keys in the expression
  // before evaluation.
  EXPECT_ANY_THROW({
    e = expr_sum_2.value(v);
    ASSERT_EQ(pt1 + pt1, e);
  });

  v.insert(k2, pt2);
  e = expr_sum_2.value(v);
  ASSERT_EQ(pt1 + pt1 + pt2, e);
}
TEST(ExpressionUsage, ArithmeticTest) {
  Key k1 = Symbol('x', 1);
  Key k2 = Symbol('x', 2);
  Point2 pt1(10, 20);
  Point2 pt2(2, 3);
  Values v;
  v.insert(k1, pt1);
  v.insert(k2, pt2);
  // Expression with keys or variables are
  // called leaf expressions.
  Expression<Point2> expr1(k1);
  Expression<Point2> expr2(k2);
  Expression<Point2> expr3(pt1/2);



  // Brackets influence how the expression tree is
  // printed. Only + operator is implemented.
  // so expr1/2 does not compile
  auto divisionby2 = [](const Point2& p, OptionalJacobian<2, 2> H){
      return p/2;
  };

  // To divide by 2 we need a create another expression
  // with unary function initialization provided.
  Expression<Point2> un_expr(divisionby2, expr1);
  Point2 e = un_expr.value(v);
  ASSERT_EQ(e, pt1/2);
  Expression<Point2> expr = expr1 + (expr2 + expr3);
  expr.print("\n");
  expr = expr1 + expr2 + expr3;
  expr.print("\n");

  // Expression multiplication and division
  // expr1 * expr2
  // We will implement a method and binary function initialization.
  // In scalar functions, this direct multiplication.
  Expression<Vector1> s1(1);
  Expression<Vector1> s2(2);
  Values v_s; v_s.insert(1,Vector1(23));v_s.insert(2,Vector1(10));
  // Number of optional Jacobians are equal to the number of
  // operands in the expression function.
  auto multiply_expr = [](const Vector1 s1, const Vector1 s2,
          OptionalJacobian<1, 1> H1, OptionalJacobian<1,1> H2) -> Vector1{
    return Vector1(s1[0]*s2[0]);
    // one can implement jacobian as well it is optional to implement.
  };
  auto division_expr = [](const Vector1 s1, const Vector1 s2,
          OptionalJacobian<1, 1> H1, OptionalJacobian<1,1> H2) -> Vector1{
    return Vector1(s1[0]/s2[0]);
    // one can implement jacobian as well it is optional to implement.
  };
  // binary function expression
  Expression<Vector1> scalar_division(division_expr, s1, s2);
  Expression<Vector1> scalar_multiplication(multiply_expr, s1, s2);
  Vector1 e_s = scalar_multiplication.value(v_s);
  ASSERT_EQ(e_s.value(), 23*10);
  e_s = scalar_division.value(v_s);
  EXPECT_NEAR(e_s.value(), 23.0f/10, 1e-4);
}

// Test predefined expressions in expressions.h
// include gtsam/nonlinear/expressions.h
TEST(ExpressionUsage, InternalDefinitionTest){
    const Key k = Symbol(1);
    // Directly giving the key as a number in
    // Double expression is ambiguous and can
    // give compiler error, Use Key type.
    typedef Expression<Point3> Point3_;
    typedef Expression<Pose3> Pose3_;
    Point3_ p(k);
    //unary expression for inbuilt function
    // norm3 is a function in gtsam namespace
    // the norm function in Point3 does not take any operands.
    Double_ norm1_expr(norm3, p);

    Values v; v.insert(k, Point3(3, 3, 3));
    auto e = norm1_expr.value(v);
    EXPECT_NEAR(e, sqrt(27.0f), 1e-4);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
