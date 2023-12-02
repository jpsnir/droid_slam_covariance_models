/* The set of tests written here are to understand different operations
 * that can be done with expression factors.
 * The main goals of these sense tests are to understand the usage of expression
 * factors in gtsam library and how auto differentiation is done in gtsam.
 * In addition, we will understand try to compare the output of expression
 * factors generated from symforce and gtsam expression factors.
 */

#include <gtest/gtest.h>
#include <gtsam/nonlinear/Expression.h>
#include <gtsam/nonlinear/ExpressionFactor.h>


using namespace gtsam;

TEST(ExpressionUsage, ConstructionTest){
}
