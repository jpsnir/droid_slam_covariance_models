#include<iostream>
#include<gtest/gtest.h>

#include<iostream>
#include<fmt/core.h>
#include<gtsam/base/Matrix.h>
#include<gtsam/3rdparty/Eigen/Eigen/Dense>
#include <gtsam/base/Vector.h>
#include<sym/pose2.h>
#include<sym/rot2.h>
#include "custom_factors/gen/cpp/symforce/sym/squared_error_factor.h"

TEST(TypeCompatibilityTest, BasicUsage){

    sym::Rot2<float> rot(M_PI/6);
    sym::Rot2<double> rot1(M_PI/6);
    Eigen::Vector2f v(0, 0);
    sym::Pose2<float> pose = {rot, v};
    Eigen::Vector2d v_g(0,0);
    gtsam::Matrix v_d(0,0);
    sym::Pose2<double> pose1 = {rot1, v_g};
    Eigen::Matrix<float,1,1> res;
    Eigen::Matrix<float,1,2> jacobian;
    Eigen::Matrix<float,2,2> hessian;
    sym::SquaredErrorFactor<float>(1, 2, 1, 2, &res, &jacobian, &hessian);
    SUCCEED();
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
