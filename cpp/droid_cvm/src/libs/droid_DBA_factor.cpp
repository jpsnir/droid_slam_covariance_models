#include <custom_factors/droid_DBA_factor.h>

namespace gtsam {

Vector DroidDBAFactor::evaluateError(
    const Pose3 &pose_i, const Pose3 &pose_j, const double &depth_i,
    boost::optional<Matrix &> H_pose_i,
    boost::optional<Matrix &> H_pose_j,
    boost::optional<Matrix &> H_depth_i) const {
  Point2 error = Point2::Zero();

  Matrix36 H_c1;
  Matrix31 H_di;
  Matrix36 H_c2;
  Matrix33 H_pt_w;
  Matrix23 H_pt_c2;
  try {
    PinholeCamera<Cal3_S2> camera1(pose_i, K_);
    Point3 backPrj_pt_w =
        camera1.backproject(pixel_i_, depth_i, H_c1, boost::none, H_di);
    Point3 pt_c2 = pose_j.transformTo(backPrj_pt_w, H_c2, H_pt_w);
    Matrix44 I = Matrix44::Identity();
    Pose3 pose_c2_c2 = Pose3(I);
    PinholeCamera<Cal3_S2> camera2(pose_c2_c2, K_);
    Point2 reprojectedPt_j = camera2.project(pt_c2, boost::none, H_pt_c2);
    // error = reprojected - measured(predicted network)
    error = reprojectedPt_j - predicted_pixel_j_;
    // Define total derivative wrt variables
    // now the function transforms the pixel points in camera i
    // to camera j.
    // So we premultiply the matrix.
    //
    if (H_pose_i)
      *H_pose_i = H_pt_c2 * H_pt_w * H_c1; // 2x3 x 3x3 x 3x6 = 2x6
    if (H_pose_j)
      *H_pose_j = H_pt_c2 * H_c2; // 2x3 x 3x6
    if (H_depth_i)
      *H_depth_i = H_pt_c2 * H_pt_w * H_di; // 2x3 x 3x3 x 3x1 = 2x1
    // Store factor is valid;
  } catch (CheiralityException &e) {
    // To make sure that the error from a point
    // from behind the camera does not impact the total error and direction of
    // derivative.
    if (H_pose_i)
      *H_pose_i = Matrix26::Zero();
    if (H_pose_j)
      *H_pose_j = Matrix26::Zero();
    if (H_depth_i)
      *H_depth_i = Matrix21::Zero();
    error = Point2(0, 0);
    std::cout << e.what() << ": Depth " << DefaultKeyFormatter(this->key3())
              << " with camera " << DefaultKeyFormatter(this->key1())
              << " has moved behind camera "
              << DefaultKeyFormatter(this->key2()) << std::endl;
    throw CheiralityException(this->key2());
  }
  return error;
}

}; // namespace gtsam
