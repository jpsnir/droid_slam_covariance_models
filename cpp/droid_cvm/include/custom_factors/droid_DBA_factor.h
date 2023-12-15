#ifndef DROID_DBA_FACTOR_H
#define DROID_DBA_FACTOR_H

#include <boost/smart_ptr/shared_ptr.hpp>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/Symbol.h>
#include <iostream>
#include <memory>

using namespace gtsam;

namespace droid_factors {
class DroidDBAFactor : public NoiseModelFactor3<Pose3, Pose3, double> {
private:
  typedef NoiseModelFactor3<Pose3, Pose3, double> Base;
  Point2 predicted_pixel_j_; // predicted measurement pixel in frame j
  Point2 pixel_i_;
  boost::shared_ptr<Cal3_S2> K_;
  bool valid_;

public:
  using Base::evaluateError;
  typedef std::shared_ptr<DroidDBAFactor> shared_ptr;
  typedef DroidDBAFactor This;

  // Default constructor
  DroidDBAFactor() : predicted_pixel_j_(0, 0){};
  ~DroidDBAFactor() override{};

  // Constructor from a ternary measurement of pose of
  // frame i, frame j and depth of pixel in frame i
  DroidDBAFactor(const Key &pi, const Key &pj, const Key &di,
                 const Point2 &pixel_i, const Point2 &predicted_pixel_j,
                 const boost::shared_ptr<Cal3_S2> &K,
                 const SharedNoiseModel &model)
      : Base(model, pi, pj, di), predicted_pixel_j_(predicted_pixel_j),
        pixel_i_(pixel_i), K_(K){};

  /* @brief evaluateError() function computes the residual
   * For this problem we are evaluating the difference between the predicted
   * pixel and the reprojected pixel. The error is in pixel space.
   * cam_i = P(T_w_c2, K)
   * e = pixel_j_predicted* - Project_cam_j(T_c2_w * cam_i.backproject(pixel_i,
   * depth)) In addition the custom factor defines the jacobians for the error
   * function.
   */
  Vector evaluateError(
      const Pose3 &pose_i, const Pose3 &pose_j, const double &depth_i,
      boost::optional<Matrix &> H_pose_i = boost::none,
      boost::optional<Matrix &> H_pose_j = boost::none,
      boost::optional<Matrix &> H_depth_i = boost::none) const override;

  //  gtsam::NonlinearFactor::shared_ptr clone() const override;
  inline const Point2 &pixelInCam_j() const { return predicted_pixel_j_; }
  inline const Point2 &pixelInCam_i() const { return pixel_i_; }
  inline const auto &calibration() const { return K_; }
  inline void setValid(bool flag) { valid_ = flag; }
  // TODO: serialization of factor
};
}; // namespace droid_factors
#endif
