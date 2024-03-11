/*
 * The tests in this file are designed to test the
 * jacobians of custom factors. Same residual error function is
 * designed using different ways in gtsam and the numerical derivative
 * is tested.
 * Gtsam implements many jacobians of its internal functions.
 */

#include <custom_factors/droid_DBA_factor.h>
#include <gtest/gtest.h>
#include <gtsam/base/Testable.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/base/utilities.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/Expression.h>
#include <numerical_derivative_py.h>
#include <vector>

using namespace gtsam;
using namespace droid_factors;
double depth1 = 1, depth = 2;
Point3 t_w_c;
std::vector<Point2> pixel_coords = {Point2(10, 60), Point2(50, 50),
                                    Point2(30, 40), Point2(30, 60)};

TEST(CustomFactorTest, BackProjectOperation)
{
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
  auto backproject2Dpoint = [&camera1, &pixel](double depth) -> Point3
  {
    return camera1.backproject(pixel, depth);
  };

  auto backproject2Dpoint1 = [&K, &pixel](double depth,
                                          Pose3 cam_pose) -> Point3
  {
    PinholeCamera<Cal3_S2> camera(cam_pose, K);
    return camera.backproject(pixel, depth);
  };

  auto backproject2Dpoint2 = [&K](Point2 pixel, double depth,
                                  Pose3 cam_pose) -> Point3
  {
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
TEST(CustomFactorTest, Transformation3DPoint)
{
  Point3 t_w_c(2, 1, 0);
  Cal3_S2 K(50, 50, 0, 50, 50);
  Rot3 R_w_c = Rot3::RzRyRx(-M_PI / 2, 0, -M_PI / 2);
  Pose3 pose_w_c1(R_w_c, t_w_c);
  t_w_c << 1.5, 1, 0;
  Pose3 pose_w_c2(R_w_c, t_w_c);
  Matrix44 T_c1_w;
  T_c1_w << 0, -1, 0, 1, 0, 0, -1, 0, 1, 0, 0, -1.5, 0, 0, 0, 1;
  Pose3 expectedPose_c2_w(T_c1_w);
  for (auto pt : pixel_coords)
  {
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
        [&pt, &K](Pose3 pose_w_ci, Pose3 pose_w_cj, double depth)
    {
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

TEST(CustomFactorTest, Pixel_i_To_j)
{
  Cal3_S2 K(50, 50, 0, 50, 50);
  Rot3 R_w_c = Rot3::RzRyRx(-M_PI / 2, 0, -M_PI / 2);
  Pose3 pose_w_c1(R_w_c, Point3(2, 1, 0));
  Pose3 pose_w_c2(R_w_c, Point3(1.5, 1, 0));
  Matrix T_c2_w = pose_w_c2.inverse().matrix();
  for (auto pt : pixel_coords)
  {
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
    Matrix actualH_pose_c2 = actualH_pt_c2 * actualH_c2; // 2x3 x 3x6
    Matrix actualH_pose_c1 =
        actualH_pt_c2 * actualH_pt_w * actualH_c1; // 2x3 x 3x3 x 3x6 = 2x6
    Matrix actualH_depth =
        actualH_pt_c2 * actualH_pt_w * actualH_d; // 2x3 x 3x3 x 3x1 = 2x1

    // Two different implementations of functions for
    // numerical derivative and the overall operation
    // to understand the composition of derivatives.
    // compare two formulations of two functions.
    auto pixel_i_to_pixel_j_f1 = [&pt, &K](Pose3 pose_w_ci, Pose3 pose_w_cj,
                                           double depth)
    {
      PinholeCamera<Cal3_S2> camera1(pose_w_ci, K);
      Point3 backPrj_pt_w = camera1.backproject(pt, depth);
      Point3 pt_c2 = pose_w_cj.transformTo(backPrj_pt_w);
      Matrix44 I = Matrix44::Identity();
      Pose3 pose_c2_c2 = Pose3(I);
      PinholeCamera<Cal3_S2> camera2(pose_c2_c2, K);
      return camera2.project(pt_c2);
    };
    auto pixel_i_to_pixel_j_f2 = [&pt, &K](Pose3 pose_w_ci, Pose3 pose_w_cj,
                                           double depth)
    {
      PinholeCamera<Cal3_S2> camera1(pose_w_ci, K);
      Point3 backPrj_pt_w = camera1.backproject(pt, depth);
      PinholeCamera<Cal3_S2> camera2(pose_w_cj, K);
      return camera2.project(backPrj_pt_w);
    };

    ASSERT_TRUE(assert_equal(
        pixel_i_to_pixel_j_f1(pose_w_c1, pose_w_c2, depth1),
        pixel_i_to_pixel_j_f2(pose_w_c1, pose_w_c2, depth1), 1e-5));

    // projection operation implemented to get pixel coordinates.
    Matrix m = K.matrix();
    Vector3 hmg_uv_c2 = m * pt_c2;
    Point2 expected_uv_c2(hmg_uv_c2.x() / hmg_uv_c2.z(),
                          hmg_uv_c2.y() / hmg_uv_c2.z());

    ASSERT_TRUE(assert_equal(uv_c2, expected_uv_c2));

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
    Matrix expectedH_d_f1 = numericalDerivative33<Point2, Pose3, Pose3, double>(
        pixel_i_to_pixel_j_f1, pose_w_c1, pose_w_c2, depth1);
    Matrix expectedH_d_f2 = numericalDerivative33<Point2, Pose3, Pose3, double>(
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

TEST(DroidDBAFactorTest, Constructor)
{
  //
  std::vector<double> depths = {1.0, 2.0};
  std::vector<Point2> pixel_coords = {Point2(10, 60), Point2(50, 50),
                                      Point2(30, 40), Point2(30, 60)};
  boost::shared_ptr<Cal3_S2> K(new Cal3_S2(50, 50, 0, 50, 50));
  Rot3 R_w_c = Rot3::RzRyRx(-M_PI / 2, 0, -M_PI / 2);
  Pose3 pose_w_c1(R_w_c, Point3(2, 1, 0));
  Pose3 pose_w_c2(R_w_c, Point3(1.5, 1, 0));
  Key d_k_1, p_k_1, p_k_2;
  d_k_1 = Symbol('d', 1);
  p_k_1 = Symbol('x', 1);
  p_k_2 = Symbol('x', 2);
  auto pixel_i = pixel_coords[0];
  auto pixel_i_to_pixel_j = [&pixel_i, &K](Pose3 pose_w_ci, Pose3 pose_w_cj,
                                           double depth)
  {
    PinholeCamera<Cal3_S2> camera1(pose_w_ci, *K);
    Point3 backPrj_pt_w = camera1.backproject(pixel_i, depth);
    PinholeCamera<Cal3_S2> camera2(pose_w_cj, *K);
    return camera2.project(backPrj_pt_w);
  };
  SharedNoiseModel pixel_noise =
      gtsam::noiseModel::Diagonal::Sigmas(Vector2(0.1, 0.1));
  Point2 predicted_pixel =
      Point2(5, 5) + pixel_i_to_pixel_j(pose_w_c1, pose_w_c2, depth1);
  auto factor = DroidDBAFactor(p_k_1, p_k_2, d_k_1, pixel_i, predicted_pixel, K,
                               pixel_noise);
  auto K_act = factor.calibration();
  auto pred_pixel_act = factor.pixelInCam_j();
  auto pixel_i_act = factor.pixelInCam_i();
  auto k = factor.keys();
  ASSERT_TRUE((*K_act).equals(*K));
  ASSERT_TRUE(assert_equal(predicted_pixel, pred_pixel_act));
  ASSERT_TRUE(assert_equal(pixel_i_act, pixel_i));
  ASSERT_EQ(k[0], p_k_1);
  ASSERT_EQ(k[1], p_k_2);
  ASSERT_EQ(k[2], d_k_1);
  auto noiseModelptr = factor.noiseModel();
  ASSERT_TRUE(assert_equal(noiseModelptr->sigmas(), pixel_noise->sigmas()));
}

TEST(DroidDBAFactorTest, evaluateError)
{
  //
  std::vector<double> depths = {1.0, 2.0};
  std::vector<Point2> pixel_coords = {Point2(10, 60), Point2(50, 50),
                                      Point2(30, 40), Point2(30, 60)};
  std::vector<Point2> errors_expected = {Point2(5, 5), Point2(12, -10),
                                         Point2(23, 32), Point2(10, 15)};
  boost::shared_ptr<Cal3_S2> K(new Cal3_S2(50, 50, 0, 50, 50));
  Rot3 R_w_c = Rot3::RzRyRx(-M_PI / 2, 0, -M_PI / 2);
  Pose3 pose_w_c1(R_w_c, Point3(2, 1, 0));
  Pose3 pose_w_c2(R_w_c, Point3(1.5, 1, 0));
  Key d_k_1, p_k_1, p_k_2;
  d_k_1 = Symbol('d', 1);
  p_k_1 = Symbol('x', 1);
  p_k_2 = Symbol('x', 2);

  for (auto depth : depths)
  {
    for (int index = 0; index < pixel_coords.size(); index++)
    {
      auto pixel_i = pixel_coords[index];
      auto error_expected = errors_expected[index];
      auto pixel_i_to_pixel_j = [&pixel_i, &K](Pose3 pose_w_ci, Pose3 pose_w_cj,
                                               double depth)
      {
        PinholeCamera<Cal3_S2> camera1(pose_w_ci, *K);
        Point3 backPrj_pt_w = camera1.backproject(pixel_i, depth);
        PinholeCamera<Cal3_S2> camera2(pose_w_cj, *K);
        return camera2.project(backPrj_pt_w);
      };
      SharedNoiseModel pixel_noise =
          gtsam::noiseModel::Diagonal::Sigmas(Vector2(0.1, 0.1));
      // error function : e = p_r - p_m , reprojected - measured
      Point2 predicted_pixel =
          pixel_i_to_pixel_j(pose_w_c1, pose_w_c2, depth) - error_expected;
      std::cout << "\n  Predicted Pixel (" << index << ")"
                << " at depth:" << depth << "\n"
                << predicted_pixel << std::endl;
      auto factor = DroidDBAFactor(p_k_1, p_k_2, d_k_1, pixel_i,
                                   predicted_pixel, K, pixel_noise);
      auto error_computed = factor.evaluateError(pose_w_c1, pose_w_c2, depth);
      ASSERT_TRUE(assert_equal(error_computed, error_expected));
    }
  }
}

TEST(DroidDBAFactorTest, evaluateErrorDerivative)
{
  //
  std::vector<double> depths = {1.0, 2.0};
  std::vector<Point2> pixel_coords = {Point2(10, 60), Point2(50, 50),
                                      Point2(30, 40), Point2(30, 60)};
  std::vector<Point2> errors_expected = {Point2(5, 5), Point2(12, -10),
                                         Point2(23, 32), Point2(10, 15)};
  boost::shared_ptr<Cal3_S2> K(new Cal3_S2(50, 50, 0, 50, 50));
  Rot3 R_w_c = Rot3::RzRyRx(-M_PI / 2, 0, -M_PI / 2);
  Pose3 pose_w_c1(R_w_c, Point3(2, 1, 0));
  Pose3 pose_w_c2(R_w_c, Point3(1.5, 1, 0));
  Key d_k_1, p_k_1, p_k_2;
  d_k_1 = Symbol('d', 1);
  p_k_1 = Symbol('x', 1);
  p_k_2 = Symbol('x', 2);

  for (auto depth : depths)
  {
    for (int index = 0; index < pixel_coords.size(); index++)
    {
      auto pixel_i = pixel_coords[index];
      auto error_expected = errors_expected[index];
      auto pixel_i_to_pixel_j = [&pixel_i, &K](Pose3 pose_w_ci, Pose3 pose_w_cj,
                                               double depth)
      {
        PinholeCamera<Cal3_S2> camera1(pose_w_ci, *K);
        Point3 backPrj_pt_w = camera1.backproject(pixel_i, depth);
        PinholeCamera<Cal3_S2> camera2(pose_w_cj, *K);
        return camera2.project(backPrj_pt_w);
      };
      SharedNoiseModel pixel_noise =
          gtsam::noiseModel::Diagonal::Sigmas(Vector2(0.1, 0.1));
      Point2 predicted_pixel =
          error_expected + pixel_i_to_pixel_j(pose_w_c1, pose_w_c2, depth1);
      auto factor = DroidDBAFactor(p_k_1, p_k_2, d_k_1, pixel_i,
                                   predicted_pixel, K, pixel_noise);
      auto compute_error_fcn = [&factor](Pose3 pose_i, Pose3 pose_j, double d)
      {
        return factor.evaluateError(pose_i, pose_j, d);
      };
      Matrix actualH_pose1, actualH_pose2, actualH_d;
      auto error_actual = factor.evaluateError(
          pose_w_c1, pose_w_c2, depth, actualH_pose1, actualH_pose2, actualH_d);
      auto expectedH_pose1 =
          numericalDerivative31<Point2, Pose3, Pose3, double>(
              compute_error_fcn, pose_w_c1, pose_w_c2, depth);
      auto expectedH_pose2 =
          numericalDerivative32<Point2, Pose3, Pose3, double>(
              compute_error_fcn, pose_w_c1, pose_w_c2, depth);
      auto expectedH_depth =
          numericalDerivative33<Point2, Pose3, Pose3, double>(
              compute_error_fcn, pose_w_c1, pose_w_c2, depth);
      ASSERT_TRUE(assert_equal(expectedH_pose1, actualH_pose1, 1e-5));
      ASSERT_TRUE(assert_equal(expectedH_pose2, actualH_pose2, 1e-5));
      ASSERT_TRUE(assert_equal(expectedH_depth, actualH_d, 1e-5));
      std::cout << "\n  H_pose1 - pixel -" << pixel_i << " : \n"
                << expectedH_pose1 << std::endl;
      std::cout << "\n  H_pose2 - pixel -" << pixel_i << " : \n"
                << expectedH_pose2 << std::endl;
      std::cout << "\n  H_depth - pixel -" << pixel_i << " : \n"
                << expectedH_depth << std::endl;
    }
  }
}

TEST(DroidDBAPy, evaluateDerivative)
{
  //
  std::vector<double> depths = {1.0, 2.0};
  std::vector<Point2> pixel_coords = {Point2(10, 60), Point2(50, 50),
                                      Point2(30, 40), Point2(30, 60)};
  std::vector<Point2> errors_expected = {Point2(5, 5), Point2(12, -10),
                                         Point2(23, 32), Point2(10, 15)};
  Vector5 K_vec;
  K_vec << 50, 50, 0, 50, 50;

  boost::shared_ptr<Cal3_S2> K(new Cal3_S2(K_vec));
  Rot3 R_w_c = Rot3::RzRyRx(-M_PI / 2, 0, -M_PI / 2);
  Pose3 pose_w_c1(R_w_c, Point3(2, 1, 0));
  Pose3 pose_w_c2(R_w_c, Point3(1.5, 1, 0));
  Key d_k_1, p_k_1, p_k_2;
  d_k_1 = Symbol('d', 1);
  p_k_1 = Symbol('x', 1);
  p_k_2 = Symbol('x', 2);

  for (auto depth : depths)
  {
    for (int index = 0; index < pixel_coords.size(); index++)
    {
      auto pixel_i = pixel_coords[index];
      auto error_expected = errors_expected[index];
      auto pixel_i_to_pixel_j = [&pixel_i, &K](Pose3 pose_w_ci, Pose3 pose_w_cj,
                                               double depth)
      {
        PinholeCamera<Cal3_S2> camera1(pose_w_ci, *K);
        Point3 backPrj_pt_w = camera1.backproject(pixel_i, depth);
        PinholeCamera<Cal3_S2> camera2(pose_w_cj, *K);
        return camera2.project(backPrj_pt_w);
      };
      Matrix22 cov;
      SharedNoiseModel pixel_noise =
          gtsam::noiseModel::Diagonal::Sigmas(Vector2(0.1, 0.2));
      cov << 0.1, 0, 0, 0.2;
      Point2 predicted_pixel =
          error_expected + pixel_i_to_pixel_j(pose_w_c1, pose_w_c2, depth1);
      Matrix num_H_pose1, num_H_pose2, num_H_d;
      Matrix44 T_1 = pose_w_c1.matrix();
      Matrix44 T_2 = pose_w_c2.matrix();

      Matrix actualH_pose1, actualH_pose2, actualH_d;
      auto factor = DroidDBAFactor(p_k_1, p_k_2, d_k_1, pixel_i,
                                   predicted_pixel, K, pixel_noise);
      auto error_actual = factor.evaluateError(
          pose_w_c1, pose_w_c2, depth, actualH_pose1, actualH_pose2, actualH_d);

      numerical_derivative_dba(T_1, T_2, depth,
                               pixel_i, predicted_pixel,
                               cov, K_vec,
                               num_H_pose1,
                               num_H_pose2, num_H_d);

      /*
      ASSERT_TRUE(assert_equal(num_H_pose1, actualH_pose1, 1e-5));
      ASSERT_TRUE(assert_equal(num_H_pose2, actualH_pose2, 1e-5));
      ASSERT_TRUE(assert_equal(num_H_d, actualH_d, 1e-5));
      */
    }
  }
}
TEST(DroidDBAFactorTest, exceptionHandlingTest)
{
  //
  auto dba_instance_fn = []()
  {
    std::vector<double> depths = {-1.0, 2.0};
    std::vector<Point2> pixel_coords = {Point2(10, 60), Point2(50, 50),
                                        Point2(30, 40), Point2(30, 60)};
    std::vector<Point2> errors_expected = {Point2(5, 5), Point2(12, -10),
                                           Point2(23, 32), Point2(10, 15)};
    boost::shared_ptr<Cal3_S2> K(new Cal3_S2(50, 50, 0, 50, 50));
    Rot3 R_w_c = Rot3::RzRyRx(-M_PI / 2, 0, -M_PI / 2);
    Pose3 pose_w_c1(R_w_c, Point3(2, 1, 0));
    Pose3 pose_w_c2(R_w_c, Point3(1.5, 1, 0));
    Key d_k_1, p_k_1, p_k_2;
    d_k_1 = Symbol('d', 1);
    p_k_1 = Symbol('x', 1);
    p_k_2 = Symbol('x', 2);
    for (auto depth : depths)
    {
      for (int index = 0; index < pixel_coords.size(); index++)
      {
        auto pixel_i = pixel_coords[index];
        auto error_expected = errors_expected[index];
        auto pixel_i_to_pixel_j =
            [&pixel_i, &K](Pose3 pose_w_ci, Pose3 pose_w_cj, double depth)
        {
          PinholeCamera<Cal3_S2> camera1(pose_w_ci, *K);
          Point3 backPrj_pt_w = camera1.backproject(pixel_i, depth);
          PinholeCamera<Cal3_S2> camera2(pose_w_cj, *K);
          return camera2.project(backPrj_pt_w);
        };
        SharedNoiseModel pixel_noise =
            gtsam::noiseModel::Diagonal::Sigmas(Vector2(0.1, 0.1));
        Point2 predicted_pixel =
            error_expected + pixel_i_to_pixel_j(pose_w_c1, pose_w_c2, depth1);
        auto factor = DroidDBAFactor(p_k_1, p_k_2, d_k_1, pixel_i,
                                     predicted_pixel, K, pixel_noise);
        auto compute_error_fcn = [&factor](Pose3 pose_i, Pose3 pose_j,
                                           double d)
        {
          return factor.evaluateError(pose_i, pose_j, d);
        };
        Matrix actualH_pose1, actualH_pose2, actualH_d;
        auto error_actual =
            factor.evaluateError(pose_w_c1, pose_w_c2, depth, actualH_pose1,
                                 actualH_pose2, actualH_d);
      }
    }
  };
  EXPECT_THROW(dba_instance_fn(), CheiralityException);
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
