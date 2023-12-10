/*
 * The tests in this file are designed to test the
 * jacobians of custom factors. Same residual error function is
 * designed using different ways in gtsam and the numerical derivative
 * is tested.
 * Gtsam implements many jacobians of its internal functions.
 */

#include <gtest/gtest.h>
#include <gtsam/base/Testable.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Expression.h>
#include <vector>
#include <custom_factors/droid_DBA_factor.h>

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

/* The Transformation3DPoint test will show how we compose the expressions
together with jacobians
// and compare it with numerical jacobians.
// The back project function has two operations defined within itself.
// 1. backproject : a) P2 x R -> R3 -      backprojection from image plane to 3d
world
//                                      in camera frame
//               b) SE(3) x R3 -> R3 -  action of SE(3) group on point in camera
//                                      to bring it in world frame
//               The combined derivative is given by the back project function
//               is given in gtsam library. We have
//               D_c1 - derivative wrt camera 1, D_d - derivative wrt depth
//               variable.
//2. Transformto : a) SE(3) x R3 -> R3 - action of SE(3) group on point in world
//                                       frame to bring it in camera frame.
//              The derivative (D_c2) wrt to pose and (D_pw) point in world
frame
//              are provided by gtsam.
//The final derivative wrt to variables is
// D_pose1 = D_c2;          3x6
// D_pose2 = D_pw*D_c1      3x6
// D_depth = D_pw*D_d       3x1
//The overall derivative of the whole set of operation has to be computed by
//chain rule and composition definitions.
//The most fundamental definition is the definition of derivative and some
//action defintions of lie groups and their derivatives given in the math.pdf.
*/
TEST(CustomFactorTest, Transformation3DPoint) {
  Point3 t_w_c(2, 1, 0);
  Cal3_S2 K(50, 50, 0, 50, 50);
  Rot3 R_w_c = Rot3::RzRyRx(-M_PI / 2, 0, -M_PI / 2);
  Pose3 pose_w_c1(R_w_c, t_w_c);
  t_w_c << 1.5, 1, 0;
  Pose3 pose_w_c2(R_w_c, t_w_c);
  Matrix44 T_c1_w;
  T_c1_w << 0, -1, 0, 1, 0, 0, -1, 0, 1, 0, 0, -1.5, 0, 0, 0, 1;
  Pose3 expectedPose_c2_w(T_c1_w);
  for (auto pt : pixel_coords) {
    PinholeCamera<Cal3_S2> camera1(pose_w_c1, K);
    Matrix36 actualH_pose;
    Matrix32 actualH_Dp;
    Matrix31 actualH_depth;
    Matrix66 actualH_pose_inverse;
    Matrix36 actualH_poseTransform;
    Matrix33 actualH_ptTransform;
    Matrix36 actualH_poseTransform_;
    Matrix33 actualH_ptTransform_;
    // point in world coordinates
    Point3 backPrj_pt_w = camera1.backproject(pt, depth1, actualH_pose,
                                              actualH_Dp, actualH_depth);
    Pose3 pose_c2_w = pose_w_c2.inverse(actualH_pose_inverse);
    ASSERT_TRUE(assert_equal(expectedPose_c2_w, pose_c2_w));
    // point in camera coordinates
    Vector4 aug;
    aug << backPrj_pt_w, 1;
    Vector hmg_coord;
    // Action on homogeneous point in 3D by SE(3) group.
    hmg_coord = pose_c2_w.matrix() * aug;
    Point3 expected_pt_c(hmg_coord[0], hmg_coord[1], hmg_coord[2]);
    Point3 pt_c2 = pose_c2_w.transformFrom(backPrj_pt_w, actualH_poseTransform,
                                           actualH_ptTransform);
    Point3 pt_c2_ = pose_w_c2.transformTo(backPrj_pt_w, actualH_poseTransform_,
                                          actualH_ptTransform_);
    ASSERT_TRUE(assert_equal(pt_c2, expected_pt_c));
    ASSERT_TRUE(assert_equal(pt_c2_, expected_pt_c));

    // Define total derivative wrt variables
    Matrix actualH_pose_c2 = actualH_poseTransform_; // 3x6
    Matrix actualH_pose_c1 =
        actualH_ptTransform_ * actualH_pose;                 // 3x3 x 3x6 = 3x6
    Matrix actualH_d = actualH_ptTransform_ * actualH_depth; // 3x3 x 3x1 = 3x1

    auto pixel_i_to_point_in_camera_j =
        [&pt, &K](Pose3 pose_w_ci, Pose3 pose_w_cj, double depth) {
          PinholeCamera<Cal3_S2> camera1(pose_w_ci, K);
          Point3 backPrj_pt_w = camera1.backproject(pt, depth);
          Point3 pt_c2 = pose_w_cj.transformTo(backPrj_pt_w);
          return pt_c2;
        };
    Matrix expectedH_pose_c1 =
        numericalDerivative31<Point3, Pose3, Pose3, double>(
            pixel_i_to_point_in_camera_j, pose_w_c1, pose_w_c2, depth1);
    ASSERT_TRUE(assert_equal(expectedH_pose_c1, actualH_pose_c1));

    Matrix expectedH_pose_c2 =
        numericalDerivative32<Point3, Pose3, Pose3, double>(
            pixel_i_to_point_in_camera_j, pose_w_c1, pose_w_c2, depth1);
    ASSERT_TRUE(assert_equal(expectedH_pose_c2, actualH_pose_c2));

    Matrix expectedH_d = numericalDerivative33<Point3, Pose3, Pose3, double>(
        pixel_i_to_point_in_camera_j, pose_w_c1, pose_w_c2, depth1);
    ASSERT_TRUE(assert_equal(expectedH_d, actualH_d));
  }
}

TEST(CustomFactorTest, Pixel_i_To_j) {

  Cal3_S2 K(50, 50, 0, 50, 50);
  Rot3 R_w_c = Rot3::RzRyRx(-M_PI / 2, 0, -M_PI / 2);
  Pose3 pose_w_c1(R_w_c, Point3(2, 1, 0));
  Pose3 pose_w_c2(R_w_c, Point3(1.5, 1, 0));
  Matrix T_c2_w = pose_w_c2.inverse().matrix();
  for (auto pt : pixel_coords) {
    PinholeCamera<Cal3_S2> camera1(pose_w_c1, K);
    Matrix36 actualH_c1;
    Matrix31 actualH_d;
    Matrix36 actualH_c2;
    Matrix33 actualH_pt_w;
    // Step 1:  point in world coordinates
    Point3 backPrj_pt_w =
        camera1.backproject(pt, depth1, actualH_c1, boost::none, actualH_d);

    // Step 2: Action on homogeneous point in 3D by SE(3) group.
    // point in camera coordinates
    Vector4 aug;
    aug << backPrj_pt_w, 1;
    Vector hmg_coord;
    hmg_coord = T_c2_w * aug;
    Point3 expected_pt_c(hmg_coord[0], hmg_coord[1], hmg_coord[2]);
    Point3 pt_c2 =
        pose_w_c2.transformTo(backPrj_pt_w, actualH_c2, actualH_pt_w);
    ASSERT_TRUE(assert_equal(pt_c2, expected_pt_c));

    // Step 3: Action of camera on 3D point, project in camera 2
    Matrix44 I = Matrix44::Identity();
    Pose3 pose_c2_c2 = Pose3(I);
    PinholeCamera<Cal3_S2> camera2(pose_c2_c2, K);
    Matrix23 actualH_pt_c2;
    Point2 uv_c2 = camera2.project(pt_c2, boost::none, actualH_pt_c2);

    // Define total derivative wrt variables
    // now the function transforms the pixel points in camera i
    // to camera j.
    // So we premultiply the matrix.
    Matrix actualH_pose_c2 = actualH_pt_c2*actualH_c2;                // 2x3 x 3x6
    Matrix actualH_pose_c1 = actualH_pt_c2*actualH_pt_w * actualH_c1; // 2x3 x 3x3 x 3x6 = 2x6
    Matrix actualH_depth = actualH_pt_c2*actualH_pt_w * actualH_d;    // 2x3 x 3x3 x 3x1 = 2x1

    // Two different implementations of functions for
    // numerical derivative and the overall operation
    // to understand the composition of derivatives.
    // compare two formulations of two functions.
    auto pixel_i_to_pixel_j_f1 = [&pt, &K](Pose3 pose_w_ci, Pose3 pose_w_cj,
                                           double depth) {
      PinholeCamera<Cal3_S2> camera1(pose_w_ci, K);
      Point3 backPrj_pt_w = camera1.backproject(pt, depth);
      Point3 pt_c2 = pose_w_cj.transformTo(backPrj_pt_w);
      Matrix44 I = Matrix44::Identity();
      Pose3 pose_c2_c2 = Pose3(I);
      PinholeCamera<Cal3_S2> camera2(pose_c2_c2, K);
      return camera2.project(pt_c2);
    };
    auto pixel_i_to_pixel_j_f2 = [&pt, &K](Pose3 pose_w_ci, Pose3 pose_w_cj,
                                           double depth) {
      PinholeCamera<Cal3_S2> camera1(pose_w_ci, K);
      Point3 backPrj_pt_w = camera1.backproject(pt, depth);
      PinholeCamera<Cal3_S2> camera2(pose_w_cj, K);
      return camera2.project(backPrj_pt_w);
    };

    ASSERT_TRUE(
        assert_equal(pixel_i_to_pixel_j_f1(pose_w_c1, pose_w_c2, depth1),
                     pixel_i_to_pixel_j_f2(pose_w_c1, pose_w_c2, depth1),
                     1e-5));

    // projection operation implemented to get pixel coordinates.
    Matrix m = K.matrix();
    Vector3 hmg_uv_c2 = m*pt_c2;
    Point2 expected_uv_c2(
            hmg_uv_c2.x()/hmg_uv_c2.z(),
            hmg_uv_c2.y()/hmg_uv_c2.z());

    ASSERT_TRUE(
            assert_equal(uv_c2, expected_uv_c2));

    // compare derivatives of two functions.
    // The assertions below prove that both functions are equivalent
    // formulations.
    // wrt pose i
    Matrix expectedH_pose_c1_f1 =
        numericalDerivative31<Point2, Pose3, Pose3, double>(
            pixel_i_to_pixel_j_f1, pose_w_c1, pose_w_c2, depth1);
    Matrix expectedH_pose_c1_f2 =
        numericalDerivative31<Point2, Pose3, Pose3, double>(
            pixel_i_to_pixel_j_f2, pose_w_c1, pose_w_c2, depth1);
    ASSERT_TRUE(assert_equal(expectedH_pose_c1_f1, expectedH_pose_c1_f2));

    // wrt pose j
    Matrix expectedH_pose_c2_f1 =
        numericalDerivative32<Point2, Pose3, Pose3, double>(
            pixel_i_to_pixel_j_f1, pose_w_c1, pose_w_c2, depth1);
    Matrix expectedH_pose_c2_f2 =
        numericalDerivative32<Point2, Pose3, Pose3, double>(
            pixel_i_to_pixel_j_f2, pose_w_c1, pose_w_c2, depth1);
    ASSERT_TRUE(assert_equal(expectedH_pose_c2_f1, expectedH_pose_c2_f2));


    // wrt depth in camera i
    Matrix expectedH_d_f1 =
        numericalDerivative33<Point2, Pose3, Pose3, double>(
            pixel_i_to_pixel_j_f1, pose_w_c1, pose_w_c2, depth1);
    Matrix expectedH_d_f2 =
        numericalDerivative33<Point2, Pose3, Pose3, double>(
            pixel_i_to_pixel_j_f2, pose_w_c1, pose_w_c2, depth1);
    ASSERT_TRUE(assert_equal(expectedH_d_f1, expectedH_d_f2));


    // Compare with actual derivatives
    //
    ASSERT_TRUE(assert_equal(expectedH_pose_c2_f1, actualH_pose_c2));
    ASSERT_TRUE(assert_equal(expectedH_pose_c2_f2, actualH_pose_c2));
    ASSERT_TRUE(assert_equal(expectedH_pose_c1_f1, actualH_pose_c1));
    ASSERT_TRUE(assert_equal(expectedH_pose_c1_f2, actualH_pose_c1));
    ASSERT_TRUE(assert_equal(expectedH_d_f1, actualH_depth));
    ASSERT_TRUE(assert_equal(expectedH_d_f2, actualH_depth));
  }
}

TEST(DroidDBAFactorTest, Constructor){

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
