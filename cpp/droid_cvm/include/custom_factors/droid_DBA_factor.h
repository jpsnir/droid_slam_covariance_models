#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/Symbol.h>
#include <iostream>
#include <memory>

using namespace gtsam;

namespace gtsam {
class DroidDBAFactor : public NoiseModelFactor3<Pose3, Pose3, double> {
private:
  typedef NoiseModelFactor3<Pose3, Pose3, double> Base;
  Point2 predicted_pixel_j_; // predicted measurement pixel in frame j
  Point2 pixel_i_;
  Cal3_S2 K_;

public:
  using Base::evaluateError;
  typedef std::shared_ptr<DroidDBAFactor> shared_ptr;
  typedef DroidDBAFactor This;

  // Default constructor
  DroidDBAFactor() : predicted_pixel_j_(0, 0){};
  ~DroidDBAFactor() override{};

  // Constructor from a ternary measurement of pose of
  // frame i, frame j and depth of pixel in frame i
  DroidDBAFactor(const Key pi, const Key pj, const Key di, const Point2 pixel_i,
                 const Point2 predicted_pixel_j, const Cal3_S2 K,
                 const SharedNoiseModel &model)
      : Base(model, pi, pj, di), predicted_pixel_j_(predicted_pixel_j),
        pixel_i_(pixel_i), K_(K){};

  /* evaluateError() function computes the residual
   * For this problem we are evaluating the difference between the predicted
   * pixel and the reprojected pixel. The error is in pixel space.
   * cam_i = P(T_w_c2, K)
   * e = pixel_j_predicted* - Project_cam_j(T_c2_w * cam_i.backproject(pixel_i,
   * depth)) In addition the custom factor defines the jacobians for the error
   * function.
   */
  Vector evaluateError(const Pose3 &pose_i, const Pose3 &pose_j,
                       const double &depth_i,
                       boost::optional<Matrix &> H_pose_i,
                       boost::optional<Matrix &> H_pose_j,
                       boost::optional<Matrix &> H_depth_i) const override;

  //  gtsam::NonlinearFactor::shared_ptr clone() const override;
  inline const Point2 &measurementIn() const { return predicted_pixel_j_; }
  inline const Point2 &pixelInCam() const { return pixel_i_; }
  inline const Cal3_S2 &Calibration() const { return K_; }

  // TODO: serialization of factor
};
}; // namespace gtsam
