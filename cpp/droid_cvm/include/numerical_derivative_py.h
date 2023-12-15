#include <gtsam/base/Testable.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <custom_factors/droid_DBA_factor.h>
#include <iostream>
using namespace gtsam;
using namespace droid_factors;
void numerical_derivative_dba(const Matrix &T_i, const Matrix &T_j,
                              const double &depth,
                              const Point2 &pixel_to_project,
                              const Point2 &predicted_pixel,
                              const Matrix22 &pixel_cov, const Vector5 &K_vec,
                              boost::optional<Matrix &> num_H_pose1,
                              boost::optional<Matrix &> num_H_pose2,
                              boost::optional<Matrix &> num_H_depth) {
  auto d_k_1 = Symbol('d', 1);
  auto p_k_1 = Symbol('x', 1);
  auto p_k_2 = Symbol('x', 2);
  boost::shared_ptr<Cal3_S2> K(new Cal3_S2(K_vec));
  Pose3 pose_i(T_i);
  Pose3 pose_j(T_j);
  auto pixel_noise = gtsam::noiseModel::Gaussian::Covariance(pixel_cov);
  auto factor = DroidDBAFactor(p_k_1, p_k_2, d_k_1, pixel_to_project,
                               predicted_pixel, K, pixel_noise);
  // lambda function
  auto compute_error_fcn = [&factor](Pose3 pose_i, Pose3 pose_j, double d) {
    return factor.evaluateError(pose_i, pose_j, d);
  };
  if (num_H_pose1)
    *num_H_pose1 = numericalDerivative31<Point2, Pose3, Pose3, double>(
        compute_error_fcn, pose_i, pose_j, depth);
  if (num_H_pose1)
    *num_H_pose2 = numericalDerivative32<Point2, Pose3, Pose3, double>(
        compute_error_fcn, pose_i, pose_j, depth);
  if (num_H_pose1)
    *num_H_depth = numericalDerivative33<Point2, Pose3, Pose3, double>(
        compute_error_fcn, pose_i, pose_j, depth);
}
