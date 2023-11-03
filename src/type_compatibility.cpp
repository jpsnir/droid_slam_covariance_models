#include<iostream>
#include<fmt/core.h>
#include<gtsam/base/Matrix.h>
#include<gtsam/3rdparty/Eigen/Eigen/Dense>
#include <gtsam/base/Vector.h>
#include<sym/pose2.h>
#include<sym/rot2.h>

int main(){
    std::cout << " Hello custom factor example " << std::endl;
    sym::Rot2<float> rot(M_PI/6);
    sym::Rot2<double> rot1(M_PI/6);
    Eigen::Vector2f v(0, 0);
    sym::Pose2<float> pose = {rot, v};
    Eigen::Vector2d v_g(0,0);
    gtsam::Matrix v_d(0,0);
    sym::Pose2<double> pose1 = {rot1, v_g};
    std::cout << pose1 << std::endl;
}
