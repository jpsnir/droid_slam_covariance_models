// -----------------------------------------------------------------------------
// This file was autogenerated by symforce from template:
//     function/FUNCTION.h.jinja
// Do NOT modify by hand.
// -----------------------------------------------------------------------------

#pragma once

#include <Eigen/Dense>

#include <sym/linear_camera_cal.h>
#include <sym/pose3.h>

namespace sym {

/**
 * residual for each pixel in the image.
 * Droid slam computes dense depth map of
 * reduce size image. The residual here is
 * just computing one single term between two images.
 *     jacobian: (1x13) jacobian of res wrt args w_pose_i (6), w_pose_j (6), d_src (1)
 *     hessian: (13x13) Gauss-Newton hessian for args w_pose_i (6), w_pose_j (6), d_src (1)
 *     rhs: (13x1) Gauss-Newton rhs for args w_pose_i (6), w_pose_j (6), d_src (1)
 */
template <typename Scalar>
void DroidSlamResidualSingleFactor(const Eigen::Matrix<Scalar, 2, 1>& dst_img_coords,
                                   const Eigen::Matrix<Scalar, 2, 1>& src_img_coords,
                                   const Scalar d_src, const sym::Pose3<Scalar>& w_pose_i,
                                   const sym::Pose3<Scalar>& w_pose_j,
                                   const sym::LinearCameraCal<Scalar>& K,
                                   const Eigen::Matrix<Scalar, 2, 1>& pixel_confidence,
                                   const Scalar epsilon,
                                   Eigen::Matrix<Scalar, 1, 1>* const res = nullptr,
                                   Eigen::Matrix<Scalar, 1, 13>* const jacobian = nullptr,
                                   Eigen::Matrix<Scalar, 13, 13>* const hessian = nullptr,
                                   Eigen::Matrix<Scalar, 13, 1>* const rhs = nullptr) {
  // Total ops: 854

  // Input arrays
  const Eigen::Matrix<Scalar, 7, 1>& _w_pose_i = w_pose_i.Data();
  const Eigen::Matrix<Scalar, 7, 1>& _w_pose_j = w_pose_j.Data();
  const Eigen::Matrix<Scalar, 4, 1>& _K = K.Data();

  // Intermediate terms (226)
  const Scalar _tmp0 = Scalar(1.0) / (pixel_confidence(0, 0));
  const Scalar _tmp1 = std::pow(_w_pose_i[2], Scalar(2));
  const Scalar _tmp2 = 2 * _tmp1;
  const Scalar _tmp3 = std::pow(_w_pose_i[1], Scalar(2));
  const Scalar _tmp4 = 2 * _tmp3 - 1;
  const Scalar _tmp5 = _tmp2 + _tmp4;
  const Scalar _tmp6 = -_tmp5;
  const Scalar _tmp7 = Scalar(1.0) / (d_src);
  const Scalar _tmp8 = _w_pose_i[3] * _w_pose_j[3];
  const Scalar _tmp9 = _w_pose_i[0] * _w_pose_j[0];
  const Scalar _tmp10 = _w_pose_i[1] * _w_pose_j[1];
  const Scalar _tmp11 = _w_pose_i[2] * _w_pose_j[2];
  const Scalar _tmp12 = _tmp10 + _tmp11 + _tmp8 + _tmp9;
  const Scalar _tmp13 = _w_pose_i[3] * _w_pose_j[1];
  const Scalar _tmp14 = _w_pose_i[0] * _w_pose_j[2];
  const Scalar _tmp15 = _w_pose_i[1] * _w_pose_j[3];
  const Scalar _tmp16 = _w_pose_i[2] * _w_pose_j[0];
  const Scalar _tmp17 = 2 * _tmp13 + 2 * _tmp14 - 2 * _tmp15 - 2 * _tmp16;
  const Scalar _tmp18 = _tmp12 * _tmp17;
  const Scalar _tmp19 = _w_pose_i[3] * _w_pose_j[2];
  const Scalar _tmp20 = _w_pose_i[1] * _w_pose_j[0];
  const Scalar _tmp21 = _w_pose_i[0] * _w_pose_j[1];
  const Scalar _tmp22 = _w_pose_i[2] * _w_pose_j[3];
  const Scalar _tmp23 = _tmp19 + _tmp20 - _tmp21 - _tmp22;
  const Scalar _tmp24 = _w_pose_i[3] * _w_pose_j[0];
  const Scalar _tmp25 = _w_pose_i[0] * _w_pose_j[3];
  const Scalar _tmp26 = _w_pose_i[1] * _w_pose_j[2];
  const Scalar _tmp27 = _w_pose_i[2] * _w_pose_j[1];
  const Scalar _tmp28 = 2 * _tmp24 - 2 * _tmp25 - 2 * _tmp26 + 2 * _tmp27;
  const Scalar _tmp29 = _tmp18 + _tmp23 * _tmp28;
  const Scalar _tmp30 = std::pow(_tmp23, Scalar(2));
  const Scalar _tmp31 = 2 * _tmp30;
  const Scalar _tmp32 = _tmp13 + _tmp14 - _tmp15 - _tmp16;
  const Scalar _tmp33 = std::pow(_tmp32, Scalar(2));
  const Scalar _tmp34 = 2 * _tmp33 - 1;
  const Scalar _tmp35 = -_tmp31 - _tmp34;
  const Scalar _tmp36 = (-_K[2] + src_img_coords(0, 0)) / _K[0];
  const Scalar _tmp37 = _tmp36 * _tmp7;
  const Scalar _tmp38 = 2 * _tmp19 + 2 * _tmp20 - 2 * _tmp21 - 2 * _tmp22;
  const Scalar _tmp39 = _tmp12 * _tmp38;
  const Scalar _tmp40 = _tmp28 * _tmp32 - _tmp39;
  const Scalar _tmp41 = (-_K[3] + src_img_coords(1, 0)) / _K[1];
  const Scalar _tmp42 = _tmp41 * _tmp7;
  const Scalar _tmp43 = 2 * _w_pose_i[3];
  const Scalar _tmp44 = _tmp43 * _w_pose_i[2];
  const Scalar _tmp45 = 2 * _w_pose_i[0] * _w_pose_i[1];
  const Scalar _tmp46 = _tmp44 + _tmp45;
  const Scalar _tmp47 = _tmp43 * _w_pose_i[1];
  const Scalar _tmp48 = 2 * _w_pose_i[2];
  const Scalar _tmp49 = _tmp48 * _w_pose_i[0];
  const Scalar _tmp50 = _tmp47 - _tmp49;
  const Scalar _tmp51 = -_tmp50;
  const Scalar _tmp52 = -_tmp46 * _w_pose_i[5] + _tmp46 * _w_pose_j[5] - _tmp51 * _w_pose_i[6] +
                        _tmp51 * _w_pose_j[6];
  const Scalar _tmp53 = _tmp29 * _tmp7 + _tmp35 * _tmp37 + _tmp40 * _tmp42 + _tmp52 -
                        _tmp6 * _w_pose_i[4] + _tmp6 * _w_pose_j[4];
  const Scalar _tmp54 = std::pow(_w_pose_i[0], Scalar(2));
  const Scalar _tmp55 = 2 * _tmp54;
  const Scalar _tmp56 = _tmp4 + _tmp55;
  const Scalar _tmp57 = -_tmp56;
  const Scalar _tmp58 = _tmp24 - _tmp25 - _tmp26 + _tmp27;
  const Scalar _tmp59 = 2 * std::pow(_tmp58, Scalar(2));
  const Scalar _tmp60 = -_tmp34 - _tmp59;
  const Scalar _tmp61 = -_tmp18 + _tmp23 * _tmp28;
  const Scalar _tmp62 = _tmp12 * _tmp28;
  const Scalar _tmp63 = _tmp17 * _tmp23 + _tmp62;
  const Scalar _tmp64 = _tmp47 + _tmp49;
  const Scalar _tmp65 = _tmp43 * _w_pose_i[0];
  const Scalar _tmp66 = _tmp48 * _w_pose_i[1];
  const Scalar _tmp67 = _tmp65 - _tmp66;
  const Scalar _tmp68 = -_tmp67;
  const Scalar _tmp69 = -_tmp64 * _w_pose_i[4] + _tmp64 * _w_pose_j[4] - _tmp68 * _w_pose_i[5] +
                        _tmp68 * _w_pose_j[5];
  const Scalar _tmp70 = _tmp37 * _tmp61 + _tmp42 * _tmp63 - _tmp57 * _w_pose_i[6] +
                        _tmp57 * _w_pose_j[6] + _tmp60 * _tmp7 + _tmp69;
  const Scalar _tmp71 = std::max<Scalar>(_tmp70, epsilon);
  const Scalar _tmp72 = Scalar(1.0) / (_tmp71);
  const Scalar _tmp73 = _K[0] * _tmp72;
  const Scalar _tmp74 = -_K[2] - _tmp53 * _tmp73 + dst_img_coords(0, 0);
  const Scalar _tmp75 = Scalar(1.0) / (pixel_confidence(1, 0));
  const Scalar _tmp76 = _tmp2 + _tmp55 - 1;
  const Scalar _tmp77 = -_tmp76;
  const Scalar _tmp78 = _tmp17 * _tmp23 - _tmp62;
  const Scalar _tmp79 = _tmp28 * _tmp32 + _tmp39;
  const Scalar _tmp80 = -_tmp31 - _tmp59 + 1;
  const Scalar _tmp81 = _tmp44 - _tmp45;
  const Scalar _tmp82 = -_tmp81;
  const Scalar _tmp83 = _tmp65 + _tmp66;
  const Scalar _tmp84 = -_tmp82 * _w_pose_i[4] + _tmp82 * _w_pose_j[4] - _tmp83 * _w_pose_i[6] +
                        _tmp83 * _w_pose_j[6];
  const Scalar _tmp85 = _tmp37 * _tmp79 + _tmp42 * _tmp80 + _tmp7 * _tmp78 - _tmp77 * _w_pose_i[5] +
                        _tmp77 * _w_pose_j[5] + _tmp84;
  const Scalar _tmp86 = _K[1] * _tmp72;
  const Scalar _tmp87 = -_K[3] - _tmp85 * _tmp86 + dst_img_coords(1, 0);
  const Scalar _tmp88 = _tmp0 * std::pow(_tmp74, Scalar(2)) + _tmp75 * std::pow(_tmp87, Scalar(2));
  const Scalar _tmp89 = -_tmp32;
  const Scalar _tmp90 = 2 * _tmp23;
  const Scalar _tmp91 = _tmp89 * _tmp90;
  const Scalar _tmp92 = _tmp23 * _tmp32;
  const Scalar _tmp93 = 2 * _tmp92;
  const Scalar _tmp94 = _tmp37 * (-_tmp91 - _tmp93);
  const Scalar _tmp95 = (Scalar(1) / Scalar(2)) * _tmp13 + (Scalar(1) / Scalar(2)) * _tmp14 -
                        Scalar(1) / Scalar(2) * _w_pose_i[1] * _w_pose_j[3] -
                        Scalar(1) / Scalar(2) * _w_pose_i[2] * _w_pose_j[0];
  const Scalar _tmp96 = -_tmp95;
  const Scalar _tmp97 = _tmp28 * _tmp96;
  const Scalar _tmp98 = -_tmp12;
  const Scalar _tmp99 = _tmp23 * _tmp98;
  const Scalar _tmp100 = _tmp97 + _tmp99;
  const Scalar _tmp101 = _tmp12 * _tmp23;
  const Scalar _tmp102 = (Scalar(1) / Scalar(2)) * _tmp24 - Scalar(1) / Scalar(2) * _tmp25 -
                         Scalar(1) / Scalar(2) * _tmp26 + (Scalar(1) / Scalar(2)) * _tmp27;
  const Scalar _tmp103 = _tmp102 * _tmp17;
  const Scalar _tmp104 = _tmp101 + _tmp103;
  const Scalar _tmp105 = _tmp12 * _tmp89;
  const Scalar _tmp106 = _tmp102 * _tmp38;
  const Scalar _tmp107 = _tmp32 * _tmp98;
  const Scalar _tmp108 = (Scalar(1) / Scalar(2)) * _tmp19 + (Scalar(1) / Scalar(2)) * _tmp20 -
                         Scalar(1) / Scalar(2) * _tmp21 - Scalar(1) / Scalar(2) * _tmp22;
  const Scalar _tmp109 = _tmp108 * _tmp28;
  const Scalar _tmp110 = _tmp107 + _tmp109;
  const Scalar _tmp111 = 2 * _tmp73;
  const Scalar _tmp112 = std::pow(_tmp71, Scalar(-2));
  const Scalar _tmp113 = Scalar(0.5) * ((((_tmp70 - epsilon) >= 0) - ((_tmp70 - epsilon) < 0)) + 1);
  const Scalar _tmp114 = -_tmp83;
  const Scalar _tmp115 = -_tmp1;
  const Scalar _tmp116 = std::pow(_w_pose_i[3], Scalar(2));
  const Scalar _tmp117 = _tmp116 - _tmp54;
  const Scalar _tmp118 = _tmp115 + _tmp117 + _tmp3;
  const Scalar _tmp119 = -_tmp118;
  const Scalar _tmp120 = 2 * _tmp58;
  const Scalar _tmp121 = _tmp120 * _tmp98;
  const Scalar _tmp122 = _tmp12 * _tmp98;
  const Scalar _tmp123 = _tmp102 * _tmp28;
  const Scalar _tmp124 = _tmp122 + _tmp123;
  const Scalar _tmp125 = _tmp17 * _tmp96;
  const Scalar _tmp126 = _tmp125 + _tmp30;
  const Scalar _tmp127 = -_tmp101 - _tmp103;
  const Scalar _tmp128 = -_tmp114 * _w_pose_i[6] + _tmp114 * _w_pose_j[6] - _tmp119 * _w_pose_i[5] +
                         _tmp119 * _w_pose_j[5] + _tmp37 * (_tmp100 + _tmp127) +
                         _tmp42 * (_tmp124 + _tmp126) + _tmp7 * (-_tmp121 - _tmp93) -
                         _tmp81 * _w_pose_i[4] + _tmp81 * _w_pose_j[4];
  const Scalar _tmp129 = _tmp0 * _tmp74;
  const Scalar _tmp130 = -_tmp3;
  const Scalar _tmp131 = _tmp1 + _tmp117 + _tmp130;
  const Scalar _tmp132 = -_tmp125 - _tmp30;
  const Scalar _tmp133 = 2 * _tmp86;
  const Scalar _tmp134 = _tmp75 * _tmp87;
  const Scalar _tmp135 =
      _tmp129 * (2 * _K[0] * _tmp112 * _tmp113 * _tmp128 * _tmp53 -
                 _tmp111 * (_tmp42 * (-_tmp105 - _tmp106 + _tmp110) + _tmp7 * (_tmp100 + _tmp104) +
                            _tmp94)) +
      _tmp134 * (2 * _K[1] * _tmp112 * _tmp113 * _tmp128 * _tmp85 -
                 _tmp133 * (-_tmp131 * _w_pose_i[6] + _tmp131 * _w_pose_j[6] +
                            _tmp37 * (_tmp105 + _tmp106 + _tmp110) + _tmp42 * (-_tmp121 - _tmp91) +
                            _tmp69 + _tmp7 * (-_tmp124 - _tmp132)));
  const Scalar _tmp136 = -_tmp23;
  const Scalar _tmp137 = _tmp120 * _tmp136;
  const Scalar _tmp138 = _tmp58 * _tmp90;
  const Scalar _tmp139 = _tmp42 * (-_tmp137 - _tmp138);
  const Scalar _tmp140 = _tmp12 * _tmp136;
  const Scalar _tmp141 = _tmp28 * _tmp95;
  const Scalar _tmp142 = _tmp103 + _tmp99;
  const Scalar _tmp143 = _tmp12 * _tmp58;
  const Scalar _tmp144 = _tmp38 * _tmp95;
  const Scalar _tmp145 = _tmp136 * _tmp32;
  const Scalar _tmp146 = (Scalar(1) / Scalar(2)) * _tmp10 + (Scalar(1) / Scalar(2)) * _tmp11 +
                         (Scalar(1) / Scalar(2)) * _tmp8 + (Scalar(1) / Scalar(2)) * _tmp9;
  const Scalar _tmp147 = -_tmp146;
  const Scalar _tmp148 = _tmp147 * _tmp28;
  const Scalar _tmp149 = _tmp145 + _tmp148;
  const Scalar _tmp150 = _tmp115 + _tmp116 + _tmp130 + _tmp54;
  const Scalar _tmp151 = 2 * _tmp107;
  const Scalar _tmp152 = _tmp17 * _tmp95;
  const Scalar _tmp153 = -_tmp123 - _tmp136 * _tmp23;
  const Scalar _tmp154 =
      -_tmp150 * _w_pose_i[4] + _tmp150 * _w_pose_j[4] + _tmp37 * (-_tmp122 - _tmp152 - _tmp153) +
      _tmp42 * (_tmp140 + _tmp141 + _tmp142) + _tmp52 + _tmp7 * (-_tmp137 - _tmp151);
  const Scalar _tmp155 = -_tmp64;
  const Scalar _tmp156 = -_tmp131;
  const Scalar _tmp157 = _tmp136 * _tmp23;
  const Scalar _tmp158 = -_tmp143;
  const Scalar _tmp159 =
      _tmp129 *
          (2 * _K[0] * _tmp112 * _tmp113 * _tmp154 * _tmp53 -
           _tmp111 * (-_tmp155 * _w_pose_i[4] + _tmp155 * _w_pose_j[4] - _tmp156 * _w_pose_i[6] +
                      _tmp156 * _w_pose_j[6] + _tmp37 * (-_tmp138 - _tmp151) +
                      _tmp42 * (-_tmp144 + _tmp149 + _tmp158) - _tmp67 * _w_pose_i[5] +
                      _tmp67 * _w_pose_j[5] + _tmp7 * (_tmp124 + _tmp152 + _tmp157))) +
      _tmp134 * (2 * _K[1] * _tmp112 * _tmp113 * _tmp154 * _tmp85 -
                 _tmp133 * (_tmp139 + _tmp37 * (_tmp143 + _tmp144 + _tmp149) +
                            _tmp7 * (-_tmp140 - _tmp141 + _tmp142)));
  const Scalar _tmp160 = -_tmp58;
  const Scalar _tmp161 = _tmp12 * _tmp160;
  const Scalar _tmp162 = _tmp108 * _tmp17;
  const Scalar _tmp163 = _tmp148 + _tmp92;
  const Scalar _tmp164 = 2 * _tmp160 * _tmp32;
  const Scalar _tmp165 = 2 * _tmp99;
  const Scalar _tmp166 = _tmp108 * _tmp38 + _tmp122;
  const Scalar _tmp167 = -_tmp102;
  const Scalar _tmp168 = _tmp167 * _tmp28;
  const Scalar _tmp169 = -_tmp168 - _tmp33;
  const Scalar _tmp170 = _K[0] * _tmp53;
  const Scalar _tmp171 = _tmp120 * _tmp32;
  const Scalar _tmp172 = _tmp7 * (-_tmp164 - _tmp171);
  const Scalar _tmp173 = _tmp160 * _tmp23;
  const Scalar _tmp174 = _tmp147 * _tmp17 + _tmp173;
  const Scalar _tmp175 = _tmp12 * _tmp32;
  const Scalar _tmp176 = _tmp109 + _tmp175;
  const Scalar _tmp177 = -_tmp161;
  const Scalar _tmp178 = 2 * _tmp112 * _tmp113;
  const Scalar _tmp179 =
      _tmp178 * (_tmp172 + _tmp37 * (-_tmp162 + _tmp163 + _tmp177) + _tmp42 * (_tmp174 + _tmp176));
  const Scalar _tmp180 = -_tmp46;
  const Scalar _tmp181 = -_tmp150;
  const Scalar _tmp182 = _tmp168 + _tmp33;
  const Scalar _tmp183 = -_tmp175;
  const Scalar _tmp184 = -_tmp109 + _tmp183;
  const Scalar _tmp185 = _K[1] * _tmp85;
  const Scalar _tmp186 =
      _tmp129 * (-_tmp111 * (-_tmp118 * _w_pose_i[5] + _tmp118 * _w_pose_j[5] +
                             _tmp37 * (-_tmp164 - _tmp165) + _tmp42 * (-_tmp166 - _tmp169) +
                             _tmp7 * (_tmp161 + _tmp162 + _tmp163) + _tmp84) +
                 _tmp170 * _tmp179) +
      _tmp134 *
          (-_tmp133 * (-_tmp180 * _w_pose_i[5] + _tmp180 * _w_pose_j[5] - _tmp181 * _w_pose_i[4] +
                       _tmp181 * _w_pose_j[4] + _tmp37 * (_tmp166 + _tmp182) +
                       _tmp42 * (-_tmp165 - _tmp171) - _tmp50 * _w_pose_i[6] +
                       _tmp50 * _w_pose_j[6] + _tmp7 * (_tmp174 + _tmp184)) +
           _tmp179 * _tmp185);
  const Scalar _tmp187 = _tmp155 * _tmp178;
  const Scalar _tmp188 = _tmp129 * (-_tmp111 * _tmp5 + _tmp170 * _tmp187) +
                         _tmp134 * (-_tmp133 * _tmp81 + _tmp185 * _tmp187);
  const Scalar _tmp189 = _tmp178 * _tmp67;
  const Scalar _tmp190 = _tmp129 * (-_tmp111 * _tmp180 + _tmp170 * _tmp189) +
                         _tmp134 * (-_tmp133 * _tmp76 + _tmp185 * _tmp189);
  const Scalar _tmp191 =
      _tmp129 * (2 * _K[0] * _tmp112 * _tmp113 * _tmp53 * _tmp56 - _tmp111 * _tmp50) +
      _tmp134 * (2 * _K[1] * _tmp112 * _tmp113 * _tmp56 * _tmp85 - _tmp114 * _tmp133);
  const Scalar _tmp192 = 2 * _tmp101;
  const Scalar _tmp193 = _tmp167 * _tmp17;
  const Scalar _tmp194 = _tmp105 + _tmp167 * _tmp38;
  const Scalar _tmp195 = 2 * _tmp143;
  const Scalar _tmp196 = std::pow(_tmp12, Scalar(2));
  const Scalar _tmp197 = _tmp168 + _tmp196;
  const Scalar _tmp198 = _tmp178 * (_tmp37 * (-_tmp193 + _tmp97) + _tmp42 * (_tmp126 + _tmp197) +
                                    _tmp7 * (-_tmp195 - _tmp93));
  const Scalar _tmp199 =
      _tmp129 * (-_tmp111 * (_tmp42 * (-_tmp184 - _tmp194) + _tmp7 * (_tmp192 + _tmp193 + _tmp97) +
                             _tmp94) +
                 _tmp170 * _tmp198) +
      _tmp134 * (-_tmp133 * (_tmp37 * (_tmp176 + _tmp194) + _tmp42 * (-_tmp195 - _tmp91) +
                             _tmp7 * (-_tmp132 - _tmp197)) +
                 _tmp185 * _tmp198);
  const Scalar _tmp200 = 2 * _tmp175;
  const Scalar _tmp201 = _tmp125 + _tmp196;
  const Scalar _tmp202 = _tmp38 * _tmp96;
  const Scalar _tmp203 = _tmp146 * _tmp28;
  const Scalar _tmp204 = _tmp145 + _tmp203;
  const Scalar _tmp205 = _tmp140 + _tmp97;
  const Scalar _tmp206 = _tmp178 * (_tmp37 * (-_tmp153 - _tmp201) + _tmp42 * (_tmp104 + _tmp205) +
                                    _tmp7 * (-_tmp137 - _tmp200));
  const Scalar _tmp207 = _tmp129 * (-_tmp111 * (_tmp37 * (-_tmp138 - _tmp200) +
                                                _tmp42 * (_tmp158 - _tmp202 + _tmp204) +
                                                _tmp7 * (_tmp123 + _tmp157 + _tmp201)) +
                                    _tmp170 * _tmp206) +
                         _tmp134 * (-_tmp133 * (_tmp139 + _tmp37 * (_tmp143 + _tmp202 + _tmp204) +
                                                _tmp7 * (-_tmp127 - _tmp205)) +
                                    _tmp185 * _tmp206);
  const Scalar _tmp208 = -_tmp108;
  const Scalar _tmp209 = _tmp196 + _tmp208 * _tmp38;
  const Scalar _tmp210 = _tmp208 * _tmp28;
  const Scalar _tmp211 = _tmp146 * _tmp17 + _tmp173;
  const Scalar _tmp212 = _tmp17 * _tmp208;
  const Scalar _tmp213 = _tmp203 + _tmp92;
  const Scalar _tmp214 = _tmp178 * (_tmp172 + _tmp37 * (_tmp177 - _tmp212 + _tmp213) +
                                    _tmp42 * (_tmp175 + _tmp210 + _tmp211));
  const Scalar _tmp215 =
      _tmp129 * (-_tmp111 * (_tmp37 * (-_tmp164 - _tmp192) + _tmp42 * (-_tmp169 - _tmp209) +
                             _tmp7 * (_tmp161 + _tmp212 + _tmp213)) +
                 _tmp170 * _tmp214) +
      _tmp134 * (-_tmp133 * (_tmp37 * (_tmp182 + _tmp209) + _tmp42 * (-_tmp171 - _tmp192) +
                             _tmp7 * (_tmp183 - _tmp210 + _tmp211)) +
                 _tmp185 * _tmp214);
  const Scalar _tmp216 = _tmp178 * _tmp64;
  const Scalar _tmp217 = _tmp129 * (-_tmp111 * _tmp6 + _tmp170 * _tmp216) +
                         _tmp134 * (-_tmp133 * _tmp82 + _tmp185 * _tmp216);
  const Scalar _tmp218 = _tmp178 * _tmp68;
  const Scalar _tmp219 = _tmp129 * (-_tmp111 * _tmp46 + _tmp170 * _tmp218) +
                         _tmp134 * (-_tmp133 * _tmp77 + _tmp185 * _tmp218);
  const Scalar _tmp220 =
      _tmp129 * (2 * _K[0] * _tmp112 * _tmp113 * _tmp53 * _tmp57 - _tmp111 * _tmp51) +
      _tmp134 * (2 * _K[1] * _tmp112 * _tmp113 * _tmp57 * _tmp85 - _tmp133 * _tmp83);
  const Scalar _tmp221 = std::pow(d_src, Scalar(-2));
  const Scalar _tmp222 = _tmp221 * _tmp36;
  const Scalar _tmp223 = _tmp221 * _tmp41;
  const Scalar _tmp224 = -_tmp221 * _tmp60 - _tmp222 * _tmp61 - _tmp223 * _tmp63;
  const Scalar _tmp225 =
      _tmp129 * (2 * _K[0] * _tmp112 * _tmp113 * _tmp224 * _tmp53 -
                 _tmp111 * (-_tmp221 * _tmp29 - _tmp222 * _tmp35 - _tmp223 * _tmp40)) +
      _tmp134 * (2 * _K[1] * _tmp112 * _tmp113 * _tmp224 * _tmp85 -
                 _tmp133 * (-_tmp221 * _tmp78 - _tmp222 * _tmp79 - _tmp223 * _tmp80));

  // Output terms (4)
  if (res != nullptr) {
    Eigen::Matrix<Scalar, 1, 1>& _res = (*res);

    _res(0, 0) = _tmp88;
  }

  if (jacobian != nullptr) {
    Eigen::Matrix<Scalar, 1, 13>& _jacobian = (*jacobian);

    _jacobian(0, 0) = _tmp135;
    _jacobian(0, 1) = _tmp159;
    _jacobian(0, 2) = _tmp186;
    _jacobian(0, 3) = _tmp188;
    _jacobian(0, 4) = _tmp190;
    _jacobian(0, 5) = _tmp191;
    _jacobian(0, 6) = _tmp199;
    _jacobian(0, 7) = _tmp207;
    _jacobian(0, 8) = _tmp215;
    _jacobian(0, 9) = _tmp217;
    _jacobian(0, 10) = _tmp219;
    _jacobian(0, 11) = _tmp220;
    _jacobian(0, 12) = _tmp225;
  }

  if (hessian != nullptr) {
    Eigen::Matrix<Scalar, 13, 13>& _hessian = (*hessian);

    _hessian(0, 0) = std::pow(_tmp135, Scalar(2));
    _hessian(1, 0) = _tmp135 * _tmp159;
    _hessian(2, 0) = _tmp135 * _tmp186;
    _hessian(3, 0) = _tmp135 * _tmp188;
    _hessian(4, 0) = _tmp135 * _tmp190;
    _hessian(5, 0) = _tmp135 * _tmp191;
    _hessian(6, 0) = _tmp135 * _tmp199;
    _hessian(7, 0) = _tmp135 * _tmp207;
    _hessian(8, 0) = _tmp135 * _tmp215;
    _hessian(9, 0) = _tmp135 * _tmp217;
    _hessian(10, 0) = _tmp135 * _tmp219;
    _hessian(11, 0) = _tmp135 * _tmp220;
    _hessian(12, 0) = _tmp135 * _tmp225;
    _hessian(0, 1) = 0;
    _hessian(1, 1) = std::pow(_tmp159, Scalar(2));
    _hessian(2, 1) = _tmp159 * _tmp186;
    _hessian(3, 1) = _tmp159 * _tmp188;
    _hessian(4, 1) = _tmp159 * _tmp190;
    _hessian(5, 1) = _tmp159 * _tmp191;
    _hessian(6, 1) = _tmp159 * _tmp199;
    _hessian(7, 1) = _tmp159 * _tmp207;
    _hessian(8, 1) = _tmp159 * _tmp215;
    _hessian(9, 1) = _tmp159 * _tmp217;
    _hessian(10, 1) = _tmp159 * _tmp219;
    _hessian(11, 1) = _tmp159 * _tmp220;
    _hessian(12, 1) = _tmp159 * _tmp225;
    _hessian(0, 2) = 0;
    _hessian(1, 2) = 0;
    _hessian(2, 2) = std::pow(_tmp186, Scalar(2));
    _hessian(3, 2) = _tmp186 * _tmp188;
    _hessian(4, 2) = _tmp186 * _tmp190;
    _hessian(5, 2) = _tmp186 * _tmp191;
    _hessian(6, 2) = _tmp186 * _tmp199;
    _hessian(7, 2) = _tmp186 * _tmp207;
    _hessian(8, 2) = _tmp186 * _tmp215;
    _hessian(9, 2) = _tmp186 * _tmp217;
    _hessian(10, 2) = _tmp186 * _tmp219;
    _hessian(11, 2) = _tmp186 * _tmp220;
    _hessian(12, 2) = _tmp186 * _tmp225;
    _hessian(0, 3) = 0;
    _hessian(1, 3) = 0;
    _hessian(2, 3) = 0;
    _hessian(3, 3) = std::pow(_tmp188, Scalar(2));
    _hessian(4, 3) = _tmp188 * _tmp190;
    _hessian(5, 3) = _tmp188 * _tmp191;
    _hessian(6, 3) = _tmp188 * _tmp199;
    _hessian(7, 3) = _tmp188 * _tmp207;
    _hessian(8, 3) = _tmp188 * _tmp215;
    _hessian(9, 3) = _tmp188 * _tmp217;
    _hessian(10, 3) = _tmp188 * _tmp219;
    _hessian(11, 3) = _tmp188 * _tmp220;
    _hessian(12, 3) = _tmp188 * _tmp225;
    _hessian(0, 4) = 0;
    _hessian(1, 4) = 0;
    _hessian(2, 4) = 0;
    _hessian(3, 4) = 0;
    _hessian(4, 4) = std::pow(_tmp190, Scalar(2));
    _hessian(5, 4) = _tmp190 * _tmp191;
    _hessian(6, 4) = _tmp190 * _tmp199;
    _hessian(7, 4) = _tmp190 * _tmp207;
    _hessian(8, 4) = _tmp190 * _tmp215;
    _hessian(9, 4) = _tmp190 * _tmp217;
    _hessian(10, 4) = _tmp190 * _tmp219;
    _hessian(11, 4) = _tmp190 * _tmp220;
    _hessian(12, 4) = _tmp190 * _tmp225;
    _hessian(0, 5) = 0;
    _hessian(1, 5) = 0;
    _hessian(2, 5) = 0;
    _hessian(3, 5) = 0;
    _hessian(4, 5) = 0;
    _hessian(5, 5) = std::pow(_tmp191, Scalar(2));
    _hessian(6, 5) = _tmp191 * _tmp199;
    _hessian(7, 5) = _tmp191 * _tmp207;
    _hessian(8, 5) = _tmp191 * _tmp215;
    _hessian(9, 5) = _tmp191 * _tmp217;
    _hessian(10, 5) = _tmp191 * _tmp219;
    _hessian(11, 5) = _tmp191 * _tmp220;
    _hessian(12, 5) = _tmp191 * _tmp225;
    _hessian(0, 6) = 0;
    _hessian(1, 6) = 0;
    _hessian(2, 6) = 0;
    _hessian(3, 6) = 0;
    _hessian(4, 6) = 0;
    _hessian(5, 6) = 0;
    _hessian(6, 6) = std::pow(_tmp199, Scalar(2));
    _hessian(7, 6) = _tmp199 * _tmp207;
    _hessian(8, 6) = _tmp199 * _tmp215;
    _hessian(9, 6) = _tmp199 * _tmp217;
    _hessian(10, 6) = _tmp199 * _tmp219;
    _hessian(11, 6) = _tmp199 * _tmp220;
    _hessian(12, 6) = _tmp199 * _tmp225;
    _hessian(0, 7) = 0;
    _hessian(1, 7) = 0;
    _hessian(2, 7) = 0;
    _hessian(3, 7) = 0;
    _hessian(4, 7) = 0;
    _hessian(5, 7) = 0;
    _hessian(6, 7) = 0;
    _hessian(7, 7) = std::pow(_tmp207, Scalar(2));
    _hessian(8, 7) = _tmp207 * _tmp215;
    _hessian(9, 7) = _tmp207 * _tmp217;
    _hessian(10, 7) = _tmp207 * _tmp219;
    _hessian(11, 7) = _tmp207 * _tmp220;
    _hessian(12, 7) = _tmp207 * _tmp225;
    _hessian(0, 8) = 0;
    _hessian(1, 8) = 0;
    _hessian(2, 8) = 0;
    _hessian(3, 8) = 0;
    _hessian(4, 8) = 0;
    _hessian(5, 8) = 0;
    _hessian(6, 8) = 0;
    _hessian(7, 8) = 0;
    _hessian(8, 8) = std::pow(_tmp215, Scalar(2));
    _hessian(9, 8) = _tmp215 * _tmp217;
    _hessian(10, 8) = _tmp215 * _tmp219;
    _hessian(11, 8) = _tmp215 * _tmp220;
    _hessian(12, 8) = _tmp215 * _tmp225;
    _hessian(0, 9) = 0;
    _hessian(1, 9) = 0;
    _hessian(2, 9) = 0;
    _hessian(3, 9) = 0;
    _hessian(4, 9) = 0;
    _hessian(5, 9) = 0;
    _hessian(6, 9) = 0;
    _hessian(7, 9) = 0;
    _hessian(8, 9) = 0;
    _hessian(9, 9) = std::pow(_tmp217, Scalar(2));
    _hessian(10, 9) = _tmp217 * _tmp219;
    _hessian(11, 9) = _tmp217 * _tmp220;
    _hessian(12, 9) = _tmp217 * _tmp225;
    _hessian(0, 10) = 0;
    _hessian(1, 10) = 0;
    _hessian(2, 10) = 0;
    _hessian(3, 10) = 0;
    _hessian(4, 10) = 0;
    _hessian(5, 10) = 0;
    _hessian(6, 10) = 0;
    _hessian(7, 10) = 0;
    _hessian(8, 10) = 0;
    _hessian(9, 10) = 0;
    _hessian(10, 10) = std::pow(_tmp219, Scalar(2));
    _hessian(11, 10) = _tmp219 * _tmp220;
    _hessian(12, 10) = _tmp219 * _tmp225;
    _hessian(0, 11) = 0;
    _hessian(1, 11) = 0;
    _hessian(2, 11) = 0;
    _hessian(3, 11) = 0;
    _hessian(4, 11) = 0;
    _hessian(5, 11) = 0;
    _hessian(6, 11) = 0;
    _hessian(7, 11) = 0;
    _hessian(8, 11) = 0;
    _hessian(9, 11) = 0;
    _hessian(10, 11) = 0;
    _hessian(11, 11) = std::pow(_tmp220, Scalar(2));
    _hessian(12, 11) = _tmp220 * _tmp225;
    _hessian(0, 12) = 0;
    _hessian(1, 12) = 0;
    _hessian(2, 12) = 0;
    _hessian(3, 12) = 0;
    _hessian(4, 12) = 0;
    _hessian(5, 12) = 0;
    _hessian(6, 12) = 0;
    _hessian(7, 12) = 0;
    _hessian(8, 12) = 0;
    _hessian(9, 12) = 0;
    _hessian(10, 12) = 0;
    _hessian(11, 12) = 0;
    _hessian(12, 12) = std::pow(_tmp225, Scalar(2));
  }

  if (rhs != nullptr) {
    Eigen::Matrix<Scalar, 13, 1>& _rhs = (*rhs);

    _rhs(0, 0) = _tmp135 * _tmp88;
    _rhs(1, 0) = _tmp159 * _tmp88;
    _rhs(2, 0) = _tmp186 * _tmp88;
    _rhs(3, 0) = _tmp188 * _tmp88;
    _rhs(4, 0) = _tmp190 * _tmp88;
    _rhs(5, 0) = _tmp191 * _tmp88;
    _rhs(6, 0) = _tmp199 * _tmp88;
    _rhs(7, 0) = _tmp207 * _tmp88;
    _rhs(8, 0) = _tmp215 * _tmp88;
    _rhs(9, 0) = _tmp217 * _tmp88;
    _rhs(10, 0) = _tmp219 * _tmp88;
    _rhs(11, 0) = _tmp220 * _tmp88;
    _rhs(12, 0) = _tmp225 * _tmp88;
  }
}  // NOLINT(readability/fn_size)

// NOLINTNEXTLINE(readability/fn_size)
}  // namespace sym