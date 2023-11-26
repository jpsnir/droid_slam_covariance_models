# -----------------------------------------------------------------------------
# This file was autogenerated by symforce from template:
#     function/FUNCTION.py.jinja
# Do NOT modify by hand.
# -----------------------------------------------------------------------------

# pylint: disable=too-many-locals,too-many-lines,too-many-statements,unused-argument,unused-import

import math
import typing as T

import numpy

import sym


def droid_slam_residual_single_factor(
    dst_img_coords, src_img_coords, d_src, w_pose_i, w_pose_j, K, pixel_confidence, epsilon
):
    # type: (numpy.ndarray, numpy.ndarray, float, sym.Pose3, sym.Pose3, sym.LinearCameraCal, numpy.ndarray, float) -> T.Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    residual for each pixel in the image.
    Droid slam computes dense depth map of
    reduce size image. The residual here is
    just computing one single term between two images.
        jacobian: (1x13) jacobian of res wrt args w_pose_i (6), w_pose_j (6), d_src (1)
        hessian: (13x13) Gauss-Newton hessian for args w_pose_i (6), w_pose_j (6), d_src (1)
        rhs: (13x1) Gauss-Newton rhs for args w_pose_i (6), w_pose_j (6), d_src (1)
    """

    # Total ops: 854

    # Input arrays
    if dst_img_coords.shape == (2,):
        dst_img_coords = dst_img_coords.reshape((2, 1))
    elif dst_img_coords.shape != (2, 1):
        raise IndexError(
            "dst_img_coords is expected to have shape (2, 1) or (2,); instead had shape {}".format(
                dst_img_coords.shape
            )
        )

    if src_img_coords.shape == (2,):
        src_img_coords = src_img_coords.reshape((2, 1))
    elif src_img_coords.shape != (2, 1):
        raise IndexError(
            "src_img_coords is expected to have shape (2, 1) or (2,); instead had shape {}".format(
                src_img_coords.shape
            )
        )

    _w_pose_i = w_pose_i.data
    _w_pose_j = w_pose_j.data
    _K = K.data
    if pixel_confidence.shape == (2,):
        pixel_confidence = pixel_confidence.reshape((2, 1))
    elif pixel_confidence.shape != (2, 1):
        raise IndexError(
            "pixel_confidence is expected to have shape (2, 1) or (2,); instead had shape {}".format(
                pixel_confidence.shape
            )
        )

    # Intermediate terms (226)
    _tmp0 = 1 / pixel_confidence[0, 0]
    _tmp1 = _w_pose_i[2] ** 2
    _tmp2 = 2 * _tmp1
    _tmp3 = _w_pose_i[1] ** 2
    _tmp4 = 2 * _tmp3 - 1
    _tmp5 = _tmp2 + _tmp4
    _tmp6 = -_tmp5
    _tmp7 = 1 / d_src
    _tmp8 = _w_pose_i[3] * _w_pose_j[3]
    _tmp9 = _w_pose_i[0] * _w_pose_j[0]
    _tmp10 = _w_pose_i[1] * _w_pose_j[1]
    _tmp11 = _w_pose_i[2] * _w_pose_j[2]
    _tmp12 = _tmp10 + _tmp11 + _tmp8 + _tmp9
    _tmp13 = _w_pose_i[3] * _w_pose_j[1]
    _tmp14 = _w_pose_i[0] * _w_pose_j[2]
    _tmp15 = _w_pose_i[1] * _w_pose_j[3]
    _tmp16 = _w_pose_i[2] * _w_pose_j[0]
    _tmp17 = 2 * _tmp13 + 2 * _tmp14 - 2 * _tmp15 - 2 * _tmp16
    _tmp18 = _tmp12 * _tmp17
    _tmp19 = _w_pose_i[3] * _w_pose_j[2]
    _tmp20 = _w_pose_i[1] * _w_pose_j[0]
    _tmp21 = _w_pose_i[0] * _w_pose_j[1]
    _tmp22 = _w_pose_i[2] * _w_pose_j[3]
    _tmp23 = _tmp19 + _tmp20 - _tmp21 - _tmp22
    _tmp24 = _w_pose_i[3] * _w_pose_j[0]
    _tmp25 = _w_pose_i[0] * _w_pose_j[3]
    _tmp26 = _w_pose_i[1] * _w_pose_j[2]
    _tmp27 = _w_pose_i[2] * _w_pose_j[1]
    _tmp28 = 2 * _tmp24 - 2 * _tmp25 - 2 * _tmp26 + 2 * _tmp27
    _tmp29 = _tmp18 + _tmp23 * _tmp28
    _tmp30 = _tmp23 ** 2
    _tmp31 = 2 * _tmp30
    _tmp32 = _tmp13 + _tmp14 - _tmp15 - _tmp16
    _tmp33 = _tmp32 ** 2
    _tmp34 = 2 * _tmp33 - 1
    _tmp35 = -_tmp31 - _tmp34
    _tmp36 = (-_K[2] + src_img_coords[0, 0]) / _K[0]
    _tmp37 = _tmp36 * _tmp7
    _tmp38 = 2 * _tmp19 + 2 * _tmp20 - 2 * _tmp21 - 2 * _tmp22
    _tmp39 = _tmp12 * _tmp38
    _tmp40 = _tmp28 * _tmp32 - _tmp39
    _tmp41 = (-_K[3] + src_img_coords[1, 0]) / _K[1]
    _tmp42 = _tmp41 * _tmp7
    _tmp43 = 2 * _w_pose_i[3]
    _tmp44 = _tmp43 * _w_pose_i[2]
    _tmp45 = 2 * _w_pose_i[0] * _w_pose_i[1]
    _tmp46 = _tmp44 + _tmp45
    _tmp47 = _tmp43 * _w_pose_i[1]
    _tmp48 = 2 * _w_pose_i[2]
    _tmp49 = _tmp48 * _w_pose_i[0]
    _tmp50 = _tmp47 - _tmp49
    _tmp51 = -_tmp50
    _tmp52 = (
        -_tmp46 * _w_pose_i[5]
        + _tmp46 * _w_pose_j[5]
        - _tmp51 * _w_pose_i[6]
        + _tmp51 * _w_pose_j[6]
    )
    _tmp53 = (
        _tmp29 * _tmp7
        + _tmp35 * _tmp37
        + _tmp40 * _tmp42
        + _tmp52
        - _tmp6 * _w_pose_i[4]
        + _tmp6 * _w_pose_j[4]
    )
    _tmp54 = _w_pose_i[0] ** 2
    _tmp55 = 2 * _tmp54
    _tmp56 = _tmp4 + _tmp55
    _tmp57 = -_tmp56
    _tmp58 = _tmp24 - _tmp25 - _tmp26 + _tmp27
    _tmp59 = 2 * _tmp58 ** 2
    _tmp60 = -_tmp34 - _tmp59
    _tmp61 = -_tmp18 + _tmp23 * _tmp28
    _tmp62 = _tmp12 * _tmp28
    _tmp63 = _tmp17 * _tmp23 + _tmp62
    _tmp64 = _tmp47 + _tmp49
    _tmp65 = _tmp43 * _w_pose_i[0]
    _tmp66 = _tmp48 * _w_pose_i[1]
    _tmp67 = _tmp65 - _tmp66
    _tmp68 = -_tmp67
    _tmp69 = (
        -_tmp64 * _w_pose_i[4]
        + _tmp64 * _w_pose_j[4]
        - _tmp68 * _w_pose_i[5]
        + _tmp68 * _w_pose_j[5]
    )
    _tmp70 = (
        _tmp37 * _tmp61
        + _tmp42 * _tmp63
        - _tmp57 * _w_pose_i[6]
        + _tmp57 * _w_pose_j[6]
        + _tmp60 * _tmp7
        + _tmp69
    )
    _tmp71 = max(_tmp70, epsilon)
    _tmp72 = 1 / _tmp71
    _tmp73 = _K[0] * _tmp72
    _tmp74 = -_K[2] - _tmp53 * _tmp73 + dst_img_coords[0, 0]
    _tmp75 = 1 / pixel_confidence[1, 0]
    _tmp76 = _tmp2 + _tmp55 - 1
    _tmp77 = -_tmp76
    _tmp78 = _tmp17 * _tmp23 - _tmp62
    _tmp79 = _tmp28 * _tmp32 + _tmp39
    _tmp80 = -_tmp31 - _tmp59 + 1
    _tmp81 = _tmp44 - _tmp45
    _tmp82 = -_tmp81
    _tmp83 = _tmp65 + _tmp66
    _tmp84 = (
        -_tmp82 * _w_pose_i[4]
        + _tmp82 * _w_pose_j[4]
        - _tmp83 * _w_pose_i[6]
        + _tmp83 * _w_pose_j[6]
    )
    _tmp85 = (
        _tmp37 * _tmp79
        + _tmp42 * _tmp80
        + _tmp7 * _tmp78
        - _tmp77 * _w_pose_i[5]
        + _tmp77 * _w_pose_j[5]
        + _tmp84
    )
    _tmp86 = _K[1] * _tmp72
    _tmp87 = -_K[3] - _tmp85 * _tmp86 + dst_img_coords[1, 0]
    _tmp88 = _tmp0 * _tmp74 ** 2 + _tmp75 * _tmp87 ** 2
    _tmp89 = -_tmp32
    _tmp90 = 2 * _tmp23
    _tmp91 = _tmp89 * _tmp90
    _tmp92 = _tmp23 * _tmp32
    _tmp93 = 2 * _tmp92
    _tmp94 = _tmp37 * (-_tmp91 - _tmp93)
    _tmp95 = (
        (1.0 / 2.0) * _tmp13
        + (1.0 / 2.0) * _tmp14
        - 1.0 / 2.0 * _w_pose_i[1] * _w_pose_j[3]
        - 1.0 / 2.0 * _w_pose_i[2] * _w_pose_j[0]
    )
    _tmp96 = -_tmp95
    _tmp97 = _tmp28 * _tmp96
    _tmp98 = -_tmp12
    _tmp99 = _tmp23 * _tmp98
    _tmp100 = _tmp97 + _tmp99
    _tmp101 = _tmp12 * _tmp23
    _tmp102 = (1.0 / 2.0) * _tmp24 - 1.0 / 2.0 * _tmp25 - 1.0 / 2.0 * _tmp26 + (1.0 / 2.0) * _tmp27
    _tmp103 = _tmp102 * _tmp17
    _tmp104 = _tmp101 + _tmp103
    _tmp105 = _tmp12 * _tmp89
    _tmp106 = _tmp102 * _tmp38
    _tmp107 = _tmp32 * _tmp98
    _tmp108 = (1.0 / 2.0) * _tmp19 + (1.0 / 2.0) * _tmp20 - 1.0 / 2.0 * _tmp21 - 1.0 / 2.0 * _tmp22
    _tmp109 = _tmp108 * _tmp28
    _tmp110 = _tmp107 + _tmp109
    _tmp111 = 2 * _tmp73
    _tmp112 = _tmp71 ** (-2)
    _tmp113 = 0.0 if (_tmp70 - epsilon) < 0 else 1.0
    _tmp114 = -_tmp83
    _tmp115 = -_tmp1
    _tmp116 = _w_pose_i[3] ** 2
    _tmp117 = _tmp116 - _tmp54
    _tmp118 = _tmp115 + _tmp117 + _tmp3
    _tmp119 = -_tmp118
    _tmp120 = 2 * _tmp58
    _tmp121 = _tmp120 * _tmp98
    _tmp122 = _tmp12 * _tmp98
    _tmp123 = _tmp102 * _tmp28
    _tmp124 = _tmp122 + _tmp123
    _tmp125 = _tmp17 * _tmp96
    _tmp126 = _tmp125 + _tmp30
    _tmp127 = -_tmp101 - _tmp103
    _tmp128 = (
        -_tmp114 * _w_pose_i[6]
        + _tmp114 * _w_pose_j[6]
        - _tmp119 * _w_pose_i[5]
        + _tmp119 * _w_pose_j[5]
        + _tmp37 * (_tmp100 + _tmp127)
        + _tmp42 * (_tmp124 + _tmp126)
        + _tmp7 * (-_tmp121 - _tmp93)
        - _tmp81 * _w_pose_i[4]
        + _tmp81 * _w_pose_j[4]
    )
    _tmp129 = _tmp0 * _tmp74
    _tmp130 = -_tmp3
    _tmp131 = _tmp1 + _tmp117 + _tmp130
    _tmp132 = -_tmp125 - _tmp30
    _tmp133 = 2 * _tmp86
    _tmp134 = _tmp75 * _tmp87
    _tmp135 = _tmp129 * (
        2 * _K[0] * _tmp112 * _tmp113 * _tmp128 * _tmp53
        - _tmp111 * (_tmp42 * (-_tmp105 - _tmp106 + _tmp110) + _tmp7 * (_tmp100 + _tmp104) + _tmp94)
    ) + _tmp134 * (
        2 * _K[1] * _tmp112 * _tmp113 * _tmp128 * _tmp85
        - _tmp133
        * (
            -_tmp131 * _w_pose_i[6]
            + _tmp131 * _w_pose_j[6]
            + _tmp37 * (_tmp105 + _tmp106 + _tmp110)
            + _tmp42 * (-_tmp121 - _tmp91)
            + _tmp69
            + _tmp7 * (-_tmp124 - _tmp132)
        )
    )
    _tmp136 = -_tmp23
    _tmp137 = _tmp120 * _tmp136
    _tmp138 = _tmp58 * _tmp90
    _tmp139 = _tmp42 * (-_tmp137 - _tmp138)
    _tmp140 = _tmp12 * _tmp136
    _tmp141 = _tmp28 * _tmp95
    _tmp142 = _tmp103 + _tmp99
    _tmp143 = _tmp12 * _tmp58
    _tmp144 = _tmp38 * _tmp95
    _tmp145 = _tmp136 * _tmp32
    _tmp146 = (
        (1.0 / 2.0) * _tmp10 + (1.0 / 2.0) * _tmp11 + (1.0 / 2.0) * _tmp8 + (1.0 / 2.0) * _tmp9
    )
    _tmp147 = -_tmp146
    _tmp148 = _tmp147 * _tmp28
    _tmp149 = _tmp145 + _tmp148
    _tmp150 = _tmp115 + _tmp116 + _tmp130 + _tmp54
    _tmp151 = 2 * _tmp107
    _tmp152 = _tmp17 * _tmp95
    _tmp153 = -_tmp123 - _tmp136 * _tmp23
    _tmp154 = (
        -_tmp150 * _w_pose_i[4]
        + _tmp150 * _w_pose_j[4]
        + _tmp37 * (-_tmp122 - _tmp152 - _tmp153)
        + _tmp42 * (_tmp140 + _tmp141 + _tmp142)
        + _tmp52
        + _tmp7 * (-_tmp137 - _tmp151)
    )
    _tmp155 = -_tmp64
    _tmp156 = -_tmp131
    _tmp157 = _tmp136 * _tmp23
    _tmp158 = -_tmp143
    _tmp159 = _tmp129 * (
        2 * _K[0] * _tmp112 * _tmp113 * _tmp154 * _tmp53
        - _tmp111
        * (
            -_tmp155 * _w_pose_i[4]
            + _tmp155 * _w_pose_j[4]
            - _tmp156 * _w_pose_i[6]
            + _tmp156 * _w_pose_j[6]
            + _tmp37 * (-_tmp138 - _tmp151)
            + _tmp42 * (-_tmp144 + _tmp149 + _tmp158)
            - _tmp67 * _w_pose_i[5]
            + _tmp67 * _w_pose_j[5]
            + _tmp7 * (_tmp124 + _tmp152 + _tmp157)
        )
    ) + _tmp134 * (
        2 * _K[1] * _tmp112 * _tmp113 * _tmp154 * _tmp85
        - _tmp133
        * (
            _tmp139
            + _tmp37 * (_tmp143 + _tmp144 + _tmp149)
            + _tmp7 * (-_tmp140 - _tmp141 + _tmp142)
        )
    )
    _tmp160 = -_tmp58
    _tmp161 = _tmp12 * _tmp160
    _tmp162 = _tmp108 * _tmp17
    _tmp163 = _tmp148 + _tmp92
    _tmp164 = 2 * _tmp160 * _tmp32
    _tmp165 = 2 * _tmp99
    _tmp166 = _tmp108 * _tmp38 + _tmp122
    _tmp167 = -_tmp102
    _tmp168 = _tmp167 * _tmp28
    _tmp169 = -_tmp168 - _tmp33
    _tmp170 = _K[0] * _tmp53
    _tmp171 = _tmp120 * _tmp32
    _tmp172 = _tmp7 * (-_tmp164 - _tmp171)
    _tmp173 = _tmp160 * _tmp23
    _tmp174 = _tmp147 * _tmp17 + _tmp173
    _tmp175 = _tmp12 * _tmp32
    _tmp176 = _tmp109 + _tmp175
    _tmp177 = -_tmp161
    _tmp178 = 2 * _tmp112 * _tmp113
    _tmp179 = _tmp178 * (
        _tmp172 + _tmp37 * (-_tmp162 + _tmp163 + _tmp177) + _tmp42 * (_tmp174 + _tmp176)
    )
    _tmp180 = -_tmp46
    _tmp181 = -_tmp150
    _tmp182 = _tmp168 + _tmp33
    _tmp183 = -_tmp175
    _tmp184 = -_tmp109 + _tmp183
    _tmp185 = _K[1] * _tmp85
    _tmp186 = _tmp129 * (
        -_tmp111
        * (
            -_tmp118 * _w_pose_i[5]
            + _tmp118 * _w_pose_j[5]
            + _tmp37 * (-_tmp164 - _tmp165)
            + _tmp42 * (-_tmp166 - _tmp169)
            + _tmp7 * (_tmp161 + _tmp162 + _tmp163)
            + _tmp84
        )
        + _tmp170 * _tmp179
    ) + _tmp134 * (
        -_tmp133
        * (
            -_tmp180 * _w_pose_i[5]
            + _tmp180 * _w_pose_j[5]
            - _tmp181 * _w_pose_i[4]
            + _tmp181 * _w_pose_j[4]
            + _tmp37 * (_tmp166 + _tmp182)
            + _tmp42 * (-_tmp165 - _tmp171)
            - _tmp50 * _w_pose_i[6]
            + _tmp50 * _w_pose_j[6]
            + _tmp7 * (_tmp174 + _tmp184)
        )
        + _tmp179 * _tmp185
    )
    _tmp187 = _tmp155 * _tmp178
    _tmp188 = _tmp129 * (-_tmp111 * _tmp5 + _tmp170 * _tmp187) + _tmp134 * (
        -_tmp133 * _tmp81 + _tmp185 * _tmp187
    )
    _tmp189 = _tmp178 * _tmp67
    _tmp190 = _tmp129 * (-_tmp111 * _tmp180 + _tmp170 * _tmp189) + _tmp134 * (
        -_tmp133 * _tmp76 + _tmp185 * _tmp189
    )
    _tmp191 = _tmp129 * (
        2 * _K[0] * _tmp112 * _tmp113 * _tmp53 * _tmp56 - _tmp111 * _tmp50
    ) + _tmp134 * (2 * _K[1] * _tmp112 * _tmp113 * _tmp56 * _tmp85 - _tmp114 * _tmp133)
    _tmp192 = 2 * _tmp101
    _tmp193 = _tmp167 * _tmp17
    _tmp194 = _tmp105 + _tmp167 * _tmp38
    _tmp195 = 2 * _tmp143
    _tmp196 = _tmp12 ** 2
    _tmp197 = _tmp168 + _tmp196
    _tmp198 = _tmp178 * (
        _tmp37 * (-_tmp193 + _tmp97) + _tmp42 * (_tmp126 + _tmp197) + _tmp7 * (-_tmp195 - _tmp93)
    )
    _tmp199 = _tmp129 * (
        -_tmp111 * (_tmp42 * (-_tmp184 - _tmp194) + _tmp7 * (_tmp192 + _tmp193 + _tmp97) + _tmp94)
        + _tmp170 * _tmp198
    ) + _tmp134 * (
        -_tmp133
        * (
            _tmp37 * (_tmp176 + _tmp194)
            + _tmp42 * (-_tmp195 - _tmp91)
            + _tmp7 * (-_tmp132 - _tmp197)
        )
        + _tmp185 * _tmp198
    )
    _tmp200 = 2 * _tmp175
    _tmp201 = _tmp125 + _tmp196
    _tmp202 = _tmp38 * _tmp96
    _tmp203 = _tmp146 * _tmp28
    _tmp204 = _tmp145 + _tmp203
    _tmp205 = _tmp140 + _tmp97
    _tmp206 = _tmp178 * (
        _tmp37 * (-_tmp153 - _tmp201) + _tmp42 * (_tmp104 + _tmp205) + _tmp7 * (-_tmp137 - _tmp200)
    )
    _tmp207 = _tmp129 * (
        -_tmp111
        * (
            _tmp37 * (-_tmp138 - _tmp200)
            + _tmp42 * (_tmp158 - _tmp202 + _tmp204)
            + _tmp7 * (_tmp123 + _tmp157 + _tmp201)
        )
        + _tmp170 * _tmp206
    ) + _tmp134 * (
        -_tmp133 * (_tmp139 + _tmp37 * (_tmp143 + _tmp202 + _tmp204) + _tmp7 * (-_tmp127 - _tmp205))
        + _tmp185 * _tmp206
    )
    _tmp208 = -_tmp108
    _tmp209 = _tmp196 + _tmp208 * _tmp38
    _tmp210 = _tmp208 * _tmp28
    _tmp211 = _tmp146 * _tmp17 + _tmp173
    _tmp212 = _tmp17 * _tmp208
    _tmp213 = _tmp203 + _tmp92
    _tmp214 = _tmp178 * (
        _tmp172 + _tmp37 * (_tmp177 - _tmp212 + _tmp213) + _tmp42 * (_tmp175 + _tmp210 + _tmp211)
    )
    _tmp215 = _tmp129 * (
        -_tmp111
        * (
            _tmp37 * (-_tmp164 - _tmp192)
            + _tmp42 * (-_tmp169 - _tmp209)
            + _tmp7 * (_tmp161 + _tmp212 + _tmp213)
        )
        + _tmp170 * _tmp214
    ) + _tmp134 * (
        -_tmp133
        * (
            _tmp37 * (_tmp182 + _tmp209)
            + _tmp42 * (-_tmp171 - _tmp192)
            + _tmp7 * (_tmp183 - _tmp210 + _tmp211)
        )
        + _tmp185 * _tmp214
    )
    _tmp216 = _tmp178 * _tmp64
    _tmp217 = _tmp129 * (-_tmp111 * _tmp6 + _tmp170 * _tmp216) + _tmp134 * (
        -_tmp133 * _tmp82 + _tmp185 * _tmp216
    )
    _tmp218 = _tmp178 * _tmp68
    _tmp219 = _tmp129 * (-_tmp111 * _tmp46 + _tmp170 * _tmp218) + _tmp134 * (
        -_tmp133 * _tmp77 + _tmp185 * _tmp218
    )
    _tmp220 = _tmp129 * (
        2 * _K[0] * _tmp112 * _tmp113 * _tmp53 * _tmp57 - _tmp111 * _tmp51
    ) + _tmp134 * (2 * _K[1] * _tmp112 * _tmp113 * _tmp57 * _tmp85 - _tmp133 * _tmp83)
    _tmp221 = d_src ** (-2)
    _tmp222 = _tmp221 * _tmp36
    _tmp223 = _tmp221 * _tmp41
    _tmp224 = -_tmp221 * _tmp60 - _tmp222 * _tmp61 - _tmp223 * _tmp63
    _tmp225 = _tmp129 * (
        2 * _K[0] * _tmp112 * _tmp113 * _tmp224 * _tmp53
        - _tmp111 * (-_tmp221 * _tmp29 - _tmp222 * _tmp35 - _tmp223 * _tmp40)
    ) + _tmp134 * (
        2 * _K[1] * _tmp112 * _tmp113 * _tmp224 * _tmp85
        - _tmp133 * (-_tmp221 * _tmp78 - _tmp222 * _tmp79 - _tmp223 * _tmp80)
    )

    # Output terms
    _res = numpy.zeros(1)
    _res[0] = _tmp88
    _jacobian = numpy.zeros(13)
    _jacobian[0] = _tmp135
    _jacobian[1] = _tmp159
    _jacobian[2] = _tmp186
    _jacobian[3] = _tmp188
    _jacobian[4] = _tmp190
    _jacobian[5] = _tmp191
    _jacobian[6] = _tmp199
    _jacobian[7] = _tmp207
    _jacobian[8] = _tmp215
    _jacobian[9] = _tmp217
    _jacobian[10] = _tmp219
    _jacobian[11] = _tmp220
    _jacobian[12] = _tmp225
    _hessian = numpy.zeros((13, 13))
    _hessian[0, 0] = _tmp135 ** 2
    _hessian[1, 0] = _tmp135 * _tmp159
    _hessian[2, 0] = _tmp135 * _tmp186
    _hessian[3, 0] = _tmp135 * _tmp188
    _hessian[4, 0] = _tmp135 * _tmp190
    _hessian[5, 0] = _tmp135 * _tmp191
    _hessian[6, 0] = _tmp135 * _tmp199
    _hessian[7, 0] = _tmp135 * _tmp207
    _hessian[8, 0] = _tmp135 * _tmp215
    _hessian[9, 0] = _tmp135 * _tmp217
    _hessian[10, 0] = _tmp135 * _tmp219
    _hessian[11, 0] = _tmp135 * _tmp220
    _hessian[12, 0] = _tmp135 * _tmp225
    _hessian[0, 1] = 0
    _hessian[1, 1] = _tmp159 ** 2
    _hessian[2, 1] = _tmp159 * _tmp186
    _hessian[3, 1] = _tmp159 * _tmp188
    _hessian[4, 1] = _tmp159 * _tmp190
    _hessian[5, 1] = _tmp159 * _tmp191
    _hessian[6, 1] = _tmp159 * _tmp199
    _hessian[7, 1] = _tmp159 * _tmp207
    _hessian[8, 1] = _tmp159 * _tmp215
    _hessian[9, 1] = _tmp159 * _tmp217
    _hessian[10, 1] = _tmp159 * _tmp219
    _hessian[11, 1] = _tmp159 * _tmp220
    _hessian[12, 1] = _tmp159 * _tmp225
    _hessian[0, 2] = 0
    _hessian[1, 2] = 0
    _hessian[2, 2] = _tmp186 ** 2
    _hessian[3, 2] = _tmp186 * _tmp188
    _hessian[4, 2] = _tmp186 * _tmp190
    _hessian[5, 2] = _tmp186 * _tmp191
    _hessian[6, 2] = _tmp186 * _tmp199
    _hessian[7, 2] = _tmp186 * _tmp207
    _hessian[8, 2] = _tmp186 * _tmp215
    _hessian[9, 2] = _tmp186 * _tmp217
    _hessian[10, 2] = _tmp186 * _tmp219
    _hessian[11, 2] = _tmp186 * _tmp220
    _hessian[12, 2] = _tmp186 * _tmp225
    _hessian[0, 3] = 0
    _hessian[1, 3] = 0
    _hessian[2, 3] = 0
    _hessian[3, 3] = _tmp188 ** 2
    _hessian[4, 3] = _tmp188 * _tmp190
    _hessian[5, 3] = _tmp188 * _tmp191
    _hessian[6, 3] = _tmp188 * _tmp199
    _hessian[7, 3] = _tmp188 * _tmp207
    _hessian[8, 3] = _tmp188 * _tmp215
    _hessian[9, 3] = _tmp188 * _tmp217
    _hessian[10, 3] = _tmp188 * _tmp219
    _hessian[11, 3] = _tmp188 * _tmp220
    _hessian[12, 3] = _tmp188 * _tmp225
    _hessian[0, 4] = 0
    _hessian[1, 4] = 0
    _hessian[2, 4] = 0
    _hessian[3, 4] = 0
    _hessian[4, 4] = _tmp190 ** 2
    _hessian[5, 4] = _tmp190 * _tmp191
    _hessian[6, 4] = _tmp190 * _tmp199
    _hessian[7, 4] = _tmp190 * _tmp207
    _hessian[8, 4] = _tmp190 * _tmp215
    _hessian[9, 4] = _tmp190 * _tmp217
    _hessian[10, 4] = _tmp190 * _tmp219
    _hessian[11, 4] = _tmp190 * _tmp220
    _hessian[12, 4] = _tmp190 * _tmp225
    _hessian[0, 5] = 0
    _hessian[1, 5] = 0
    _hessian[2, 5] = 0
    _hessian[3, 5] = 0
    _hessian[4, 5] = 0
    _hessian[5, 5] = _tmp191 ** 2
    _hessian[6, 5] = _tmp191 * _tmp199
    _hessian[7, 5] = _tmp191 * _tmp207
    _hessian[8, 5] = _tmp191 * _tmp215
    _hessian[9, 5] = _tmp191 * _tmp217
    _hessian[10, 5] = _tmp191 * _tmp219
    _hessian[11, 5] = _tmp191 * _tmp220
    _hessian[12, 5] = _tmp191 * _tmp225
    _hessian[0, 6] = 0
    _hessian[1, 6] = 0
    _hessian[2, 6] = 0
    _hessian[3, 6] = 0
    _hessian[4, 6] = 0
    _hessian[5, 6] = 0
    _hessian[6, 6] = _tmp199 ** 2
    _hessian[7, 6] = _tmp199 * _tmp207
    _hessian[8, 6] = _tmp199 * _tmp215
    _hessian[9, 6] = _tmp199 * _tmp217
    _hessian[10, 6] = _tmp199 * _tmp219
    _hessian[11, 6] = _tmp199 * _tmp220
    _hessian[12, 6] = _tmp199 * _tmp225
    _hessian[0, 7] = 0
    _hessian[1, 7] = 0
    _hessian[2, 7] = 0
    _hessian[3, 7] = 0
    _hessian[4, 7] = 0
    _hessian[5, 7] = 0
    _hessian[6, 7] = 0
    _hessian[7, 7] = _tmp207 ** 2
    _hessian[8, 7] = _tmp207 * _tmp215
    _hessian[9, 7] = _tmp207 * _tmp217
    _hessian[10, 7] = _tmp207 * _tmp219
    _hessian[11, 7] = _tmp207 * _tmp220
    _hessian[12, 7] = _tmp207 * _tmp225
    _hessian[0, 8] = 0
    _hessian[1, 8] = 0
    _hessian[2, 8] = 0
    _hessian[3, 8] = 0
    _hessian[4, 8] = 0
    _hessian[5, 8] = 0
    _hessian[6, 8] = 0
    _hessian[7, 8] = 0
    _hessian[8, 8] = _tmp215 ** 2
    _hessian[9, 8] = _tmp215 * _tmp217
    _hessian[10, 8] = _tmp215 * _tmp219
    _hessian[11, 8] = _tmp215 * _tmp220
    _hessian[12, 8] = _tmp215 * _tmp225
    _hessian[0, 9] = 0
    _hessian[1, 9] = 0
    _hessian[2, 9] = 0
    _hessian[3, 9] = 0
    _hessian[4, 9] = 0
    _hessian[5, 9] = 0
    _hessian[6, 9] = 0
    _hessian[7, 9] = 0
    _hessian[8, 9] = 0
    _hessian[9, 9] = _tmp217 ** 2
    _hessian[10, 9] = _tmp217 * _tmp219
    _hessian[11, 9] = _tmp217 * _tmp220
    _hessian[12, 9] = _tmp217 * _tmp225
    _hessian[0, 10] = 0
    _hessian[1, 10] = 0
    _hessian[2, 10] = 0
    _hessian[3, 10] = 0
    _hessian[4, 10] = 0
    _hessian[5, 10] = 0
    _hessian[6, 10] = 0
    _hessian[7, 10] = 0
    _hessian[8, 10] = 0
    _hessian[9, 10] = 0
    _hessian[10, 10] = _tmp219 ** 2
    _hessian[11, 10] = _tmp219 * _tmp220
    _hessian[12, 10] = _tmp219 * _tmp225
    _hessian[0, 11] = 0
    _hessian[1, 11] = 0
    _hessian[2, 11] = 0
    _hessian[3, 11] = 0
    _hessian[4, 11] = 0
    _hessian[5, 11] = 0
    _hessian[6, 11] = 0
    _hessian[7, 11] = 0
    _hessian[8, 11] = 0
    _hessian[9, 11] = 0
    _hessian[10, 11] = 0
    _hessian[11, 11] = _tmp220 ** 2
    _hessian[12, 11] = _tmp220 * _tmp225
    _hessian[0, 12] = 0
    _hessian[1, 12] = 0
    _hessian[2, 12] = 0
    _hessian[3, 12] = 0
    _hessian[4, 12] = 0
    _hessian[5, 12] = 0
    _hessian[6, 12] = 0
    _hessian[7, 12] = 0
    _hessian[8, 12] = 0
    _hessian[9, 12] = 0
    _hessian[10, 12] = 0
    _hessian[11, 12] = 0
    _hessian[12, 12] = _tmp225 ** 2
    _rhs = numpy.zeros(13)
    _rhs[0] = _tmp135 * _tmp88
    _rhs[1] = _tmp159 * _tmp88
    _rhs[2] = _tmp186 * _tmp88
    _rhs[3] = _tmp188 * _tmp88
    _rhs[4] = _tmp190 * _tmp88
    _rhs[5] = _tmp191 * _tmp88
    _rhs[6] = _tmp199 * _tmp88
    _rhs[7] = _tmp207 * _tmp88
    _rhs[8] = _tmp215 * _tmp88
    _rhs[9] = _tmp217 * _tmp88
    _rhs[10] = _tmp219 * _tmp88
    _rhs[11] = _tmp220 * _tmp88
    _rhs[12] = _tmp225 * _tmp88
    return _res, _jacobian, _hessian, _rhs
