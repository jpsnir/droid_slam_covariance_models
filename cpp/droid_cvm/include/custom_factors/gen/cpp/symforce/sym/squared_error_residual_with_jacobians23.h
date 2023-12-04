// -----------------------------------------------------------------------------
// This file was autogenerated by symforce from template:
//     function/FUNCTION.h.jinja
// Do NOT modify by hand.
// -----------------------------------------------------------------------------

#pragma once

#include <Eigen/Dense>

namespace sym {

/**
 * squared distance error for a single variable
 * linear regression problem
 * y = ax + b
 *
 * e = (y - a*x - b))^2
 *     res_D_a: (1x1) jacobian of res (1) wrt arg a (1)
 *     res_D_b: (1x1) jacobian of res (1) wrt arg b (1)
 */
template <typename Scalar>
Eigen::Matrix<Scalar, 1, 1> SquaredErrorResidualWithJacobians23(
    const Scalar x, const Scalar y, const Scalar a, const Scalar b,
    Eigen::Matrix<Scalar, 1, 1>* const res_D_a = nullptr,
    Eigen::Matrix<Scalar, 1, 1>* const res_D_b = nullptr) {
  // Total ops: 12

  // Input arrays

  // Intermediate terms (2)
  const Scalar _tmp0 = a * x;
  const Scalar _tmp1 = -_tmp0 - b + y;

  // Output terms (3)
  Eigen::Matrix<Scalar, 1, 1> _res;

  _res(0, 0) = std::pow(_tmp1, Scalar(2));

  if (res_D_a != nullptr) {
    Eigen::Matrix<Scalar, 1, 1>& _res_D_a = (*res_D_a);

    _res_D_a(0, 0) = -2 * _tmp1 * x;
  }

  if (res_D_b != nullptr) {
    Eigen::Matrix<Scalar, 1, 1>& _res_D_b = (*res_D_b);

    _res_D_b(0, 0) = 2 * _tmp0 + 2 * b - 2 * y;
  }

  return _res;
}  // NOLINT(readability/fn_size)

// NOLINTNEXTLINE(readability/fn_size)
}  // namespace sym