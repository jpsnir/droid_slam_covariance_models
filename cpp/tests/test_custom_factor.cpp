/*
 * The tests in this file are designed to test the
 * jacobians of custom factors. Same residual error function is
 * designed using different ways in gtsam and the numerical derivative
 * is tested.
 * Gtsam implements many jacobians of its internal functions.
 */

#include <gtest/gtest.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Expression.h>
#include <vector>
#include <gtsam/base/Testable.h>
using namespace gtsam;

double depth1 = 1, depth = 2;
Point3 t_w_c;
std::vector<Point2> pixel_coords = {Point2(10, 60), Point2(50, 50),
                                    Point2(30, 40), Point2(30, 60)};

TEST(CustomFactorTest, BackProjectOperation) {
  Point3 t_w_c(2, 1, 0);
  Cal3_S2 K(50, 50, 0, 50, 50);
  Rot3 R_w_c = Rot3::RzRyRx(-M_PI / 2, 0, -M_PI / 2);
  Pose3 pose_w_c1(R_w_c, t_w_c);
  PinholeCamera<Cal3_S2> camera1(pose_w_c1, K);
  Matrix36 actualH_pose;
  Matrix32 actualH_Dp;
  Matrix31 actualH_depth;
  Point3 backPrj_pt = camera1.backproject(pixel_coords[0], depth1, actualH_pose,
                                          actualH_Dp, actualH_depth);
  Point3 expectedPt(3, 1.8, -0.2);
  Point2 pixel = pixel_coords[0];
  auto backproject2Dpoint = [&camera1, &pixel](double depth) -> Point3 {
    return camera1.backproject(pixel, depth);
  };

  auto backproject2Dpoint1 = [&K, &pixel](double depth,
                                          Pose3 cam_pose) -> Point3 {
    PinholeCamera<Cal3_S2> camera(cam_pose, K);
    return camera.backproject(pixel, depth);
  };

  auto backproject2Dpoint2 = [&K](Point2 pixel, double depth,
                                  Pose3 cam_pose) -> Point3 {
    PinholeCamera<Cal3_S2> camera(cam_pose, K);
    return camera.backproject(pixel, depth);
  };
  Matrix expectedH =
      numericalDerivative11<Point3, double>(backproject2Dpoint, depth1);
  ASSERT_TRUE(actualH_depth.isApprox(expectedH, 1e-5));
  Matrix expectedH1_depth = numericalDerivative21<Point3, double, Pose3>(
      backproject2Dpoint1, depth1, pose_w_c1);
  ASSERT_TRUE(actualH_depth.isApprox(expectedH1_depth, 1e-5));
  ASSERT_TRUE(expectedH.isApprox(expectedH1_depth, 1e-5));
  Matrix expectedH1_pose = numericalDerivative22<Point3, double, Pose3>(
      backproject2Dpoint1, depth1, pose_w_c1);
  ASSERT_TRUE(actualH_pose.isApprox(expectedH1_pose, 1e-5));
  Matrix expectedH2_pixel =
      numericalDerivative31<Point3, Point2, double, Pose3>(
          backproject2Dpoint2, pixel_coords[0], depth1, pose_w_c1);
  ASSERT_TRUE(actualH_Dp.isApprox(expectedH2_pixel, 1e-5));
  ASSERT_TRUE(assert_equal(expectedPt[0], backPrj_pt[0], 1e-9));
}

TEST(CustomFactorTest, Transformation3DPoint) {
  Point3 t_w_c(2, 1, 0);
  Cal3_S2 K(50, 50, 0, 50, 50);
  Rot3 R_w_c = Rot3::RzRyRx(-M_PI / 2, 0, -M_PI / 2);
  Pose3 pose_w_c1(R_w_c, t_w_c);
  t_w_c << 1.5, 1, 0;
  Pose3 pose_w_c2(R_w_c, t_w_c);
  PinholeCamera<Cal3_S2> camera1(pose_w_c1, K);
  Matrix36 actualH_pose;
  Matrix32 actualH_Dp;
  Matrix31 actualH_depth;
  Matrix66 actualH_pose_inverse;
  Matrix36 actualH_poseTransform;
  Matrix33 actualH_ptTransform;
  Point3 backPrj_pt = camera1.backproject(pixel_coords[0], depth1, actualH_pose,
                                          actualH_Dp, actualH_depth);
  Pose3 pose_c2_w = pose_w_c2.inverse(actualH_pose_inverse);
  std::cout << "\n Pose Inverse Jacobian :\n " << actualH_pose_inverse << std::endl;
  Point3 tf_point = pose_c2_w.transformFrom(
          backPrj_pt, actualH_poseTransform, actualH_ptTransform);

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
