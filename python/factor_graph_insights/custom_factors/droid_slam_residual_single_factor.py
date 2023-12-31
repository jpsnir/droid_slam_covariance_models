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
    dst_img_coords, src_img_coords, d_src, w_pose_i, w_pose_j, K, epsilon
):
    # type: (numpy.ndarray, numpy.ndarray, float, sym.Pose3, sym.Pose3, sym.LinearCameraCal, float) -> T.Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    residual for each pixel in the image.
    Droid slam computes dense depth map of
    reduce size image. The residual here is
    just computing one single term between two images.
        jacobian: (2x13) jacobian of res wrt args w_pose_i (6), w_pose_j (6), d_src (1)
        hessian: (13x13) Gauss-Newton hessian for args w_pose_i (6), w_pose_j (6), d_src (1)
        rhs: (13x1) Gauss-Newton rhs for args w_pose_i (6), w_pose_j (6), d_src (1)
    """

    # Total ops: 1001

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

    # Intermediate terms (232)
    _tmp0 = _w_pose_i[2] ** 2
    _tmp1 = 2 * _tmp0
    _tmp2 = _w_pose_i[1] ** 2
    _tmp3 = 2 * _tmp2 - 1
    _tmp4 = _tmp1 + _tmp3
    _tmp5 = -_tmp4
    _tmp6 = 1 / d_src
    _tmp7 = _w_pose_i[3] * _w_pose_j[3]
    _tmp8 = _w_pose_i[0] * _w_pose_j[0]
    _tmp9 = _w_pose_i[1] * _w_pose_j[1]
    _tmp10 = _w_pose_i[2] * _w_pose_j[2]
    _tmp11 = _tmp10 + _tmp7 + _tmp8 + _tmp9
    _tmp12 = _w_pose_i[3] * _w_pose_j[1]
    _tmp13 = _w_pose_i[0] * _w_pose_j[2]
    _tmp14 = _w_pose_i[1] * _w_pose_j[3]
    _tmp15 = _w_pose_i[2] * _w_pose_j[0]
    _tmp16 = 2 * _tmp12 + 2 * _tmp13 - 2 * _tmp14 - 2 * _tmp15
    _tmp17 = _tmp11 * _tmp16
    _tmp18 = _w_pose_i[3] * _w_pose_j[2]
    _tmp19 = _w_pose_i[1] * _w_pose_j[0]
    _tmp20 = _w_pose_i[0] * _w_pose_j[1]
    _tmp21 = _w_pose_i[2] * _w_pose_j[3]
    _tmp22 = _tmp18 + _tmp19 - _tmp20 - _tmp21
    _tmp23 = _w_pose_i[3] * _w_pose_j[0]
    _tmp24 = _w_pose_i[0] * _w_pose_j[3]
    _tmp25 = _w_pose_i[1] * _w_pose_j[2]
    _tmp26 = _w_pose_i[2] * _w_pose_j[1]
    _tmp27 = 2 * _tmp23 - 2 * _tmp24 - 2 * _tmp25 + 2 * _tmp26
    _tmp28 = _tmp17 + _tmp22 * _tmp27
    _tmp29 = _tmp22 ** 2
    _tmp30 = 2 * _tmp29
    _tmp31 = _tmp12 + _tmp13 - _tmp14 - _tmp15
    _tmp32 = _tmp31 ** 2
    _tmp33 = 2 * _tmp32 - 1
    _tmp34 = -_tmp30 - _tmp33
    _tmp35 = (-_K[2] + src_img_coords[0, 0]) / _K[0]
    _tmp36 = _tmp35 * _tmp6
    _tmp37 = 2 * _tmp18 + 2 * _tmp19 - 2 * _tmp20 - 2 * _tmp21
    _tmp38 = _tmp11 * _tmp37
    _tmp39 = _tmp27 * _tmp31 - _tmp38
    _tmp40 = (-_K[3] + src_img_coords[1, 0]) / _K[1]
    _tmp41 = _tmp40 * _tmp6
    _tmp42 = 2 * _w_pose_i[3]
    _tmp43 = _tmp42 * _w_pose_i[2]
    _tmp44 = 2 * _w_pose_i[0] * _w_pose_i[1]
    _tmp45 = _tmp43 + _tmp44
    _tmp46 = _tmp42 * _w_pose_i[1]
    _tmp47 = 2 * _w_pose_i[2]
    _tmp48 = _tmp47 * _w_pose_i[0]
    _tmp49 = _tmp46 - _tmp48
    _tmp50 = -_tmp49
    _tmp51 = (
        -_tmp45 * _w_pose_i[5]
        + _tmp45 * _w_pose_j[5]
        - _tmp50 * _w_pose_i[6]
        + _tmp50 * _w_pose_j[6]
    )
    _tmp52 = (
        _tmp28 * _tmp6
        + _tmp34 * _tmp36
        + _tmp39 * _tmp41
        - _tmp5 * _w_pose_i[4]
        + _tmp5 * _w_pose_j[4]
        + _tmp51
    )
    _tmp53 = _w_pose_i[0] ** 2
    _tmp54 = 2 * _tmp53
    _tmp55 = _tmp3 + _tmp54
    _tmp56 = -_tmp55
    _tmp57 = _tmp23 - _tmp24 - _tmp25 + _tmp26
    _tmp58 = 2 * _tmp57 ** 2
    _tmp59 = -_tmp33 - _tmp58
    _tmp60 = -_tmp17 + _tmp22 * _tmp27
    _tmp61 = _tmp11 * _tmp27
    _tmp62 = _tmp16 * _tmp22 + _tmp61
    _tmp63 = _tmp46 + _tmp48
    _tmp64 = _tmp42 * _w_pose_i[0]
    _tmp65 = _tmp47 * _w_pose_i[1]
    _tmp66 = _tmp64 - _tmp65
    _tmp67 = -_tmp66
    _tmp68 = (
        -_tmp63 * _w_pose_i[4]
        + _tmp63 * _w_pose_j[4]
        - _tmp67 * _w_pose_i[5]
        + _tmp67 * _w_pose_j[5]
    )
    _tmp69 = (
        _tmp36 * _tmp60
        + _tmp41 * _tmp62
        - _tmp56 * _w_pose_i[6]
        + _tmp56 * _w_pose_j[6]
        + _tmp59 * _tmp6
        + _tmp68
    )
    _tmp70 = max(_tmp69, epsilon)
    _tmp71 = 1 / _tmp70
    _tmp72 = _K[0] * _tmp71
    _tmp73 = -_K[2] - _tmp52 * _tmp72 + dst_img_coords[0, 0]
    _tmp74 = _tmp1 + _tmp54 - 1
    _tmp75 = -_tmp74
    _tmp76 = _tmp16 * _tmp22 - _tmp61
    _tmp77 = _tmp27 * _tmp31 + _tmp38
    _tmp78 = -_tmp30 - _tmp58 + 1
    _tmp79 = _tmp43 - _tmp44
    _tmp80 = -_tmp79
    _tmp81 = _tmp64 + _tmp65
    _tmp82 = (
        -_tmp80 * _w_pose_i[4]
        + _tmp80 * _w_pose_j[4]
        - _tmp81 * _w_pose_i[6]
        + _tmp81 * _w_pose_j[6]
    )
    _tmp83 = (
        _tmp36 * _tmp77
        + _tmp41 * _tmp78
        + _tmp6 * _tmp76
        - _tmp75 * _w_pose_i[5]
        + _tmp75 * _w_pose_j[5]
        + _tmp82
    )
    _tmp84 = _K[1] * _tmp71
    _tmp85 = -_K[3] - _tmp83 * _tmp84 + dst_img_coords[1, 0]
    _tmp86 = -_tmp31
    _tmp87 = 2 * _tmp22
    _tmp88 = _tmp86 * _tmp87
    _tmp89 = _tmp22 * _tmp31
    _tmp90 = 2 * _tmp89
    _tmp91 = _tmp36 * (-_tmp88 - _tmp90)
    _tmp92 = (
        (1.0 / 2.0) * _tmp12
        + (1.0 / 2.0) * _tmp13
        - 1.0 / 2.0 * _w_pose_i[1] * _w_pose_j[3]
        - 1.0 / 2.0 * _w_pose_i[2] * _w_pose_j[0]
    )
    _tmp93 = -_tmp92
    _tmp94 = _tmp27 * _tmp93
    _tmp95 = -_tmp11
    _tmp96 = _tmp22 * _tmp95
    _tmp97 = _tmp94 + _tmp96
    _tmp98 = _tmp11 * _tmp22
    _tmp99 = (1.0 / 2.0) * _tmp23 - 1.0 / 2.0 * _tmp24 - 1.0 / 2.0 * _tmp25 + (1.0 / 2.0) * _tmp26
    _tmp100 = _tmp16 * _tmp99
    _tmp101 = _tmp100 + _tmp98
    _tmp102 = _tmp11 * _tmp86
    _tmp103 = _tmp37 * _tmp99
    _tmp104 = _tmp31 * _tmp95
    _tmp105 = (1.0 / 2.0) * _tmp18 + (1.0 / 2.0) * _tmp19 - 1.0 / 2.0 * _tmp20 - 1.0 / 2.0 * _tmp21
    _tmp106 = _tmp105 * _tmp27
    _tmp107 = _tmp104 + _tmp106
    _tmp108 = _tmp70 ** (-2)
    _tmp109 = 0.0 if (_tmp69 - epsilon) < 0 else 1.0
    _tmp110 = -_tmp81
    _tmp111 = -_tmp0
    _tmp112 = _w_pose_i[3] ** 2
    _tmp113 = _tmp112 - _tmp53
    _tmp114 = _tmp111 + _tmp113 + _tmp2
    _tmp115 = -_tmp114
    _tmp116 = 2 * _tmp57
    _tmp117 = _tmp116 * _tmp95
    _tmp118 = _tmp11 * _tmp95
    _tmp119 = _tmp27 * _tmp99
    _tmp120 = _tmp118 + _tmp119
    _tmp121 = _tmp16 * _tmp93
    _tmp122 = _tmp121 + _tmp29
    _tmp123 = -_tmp100 - _tmp98
    _tmp124 = (
        -_tmp110 * _w_pose_i[6]
        + _tmp110 * _w_pose_j[6]
        - _tmp115 * _w_pose_i[5]
        + _tmp115 * _w_pose_j[5]
        + _tmp36 * (_tmp123 + _tmp97)
        + _tmp41 * (_tmp120 + _tmp122)
        + _tmp6 * (-_tmp117 - _tmp90)
        - _tmp79 * _w_pose_i[4]
        + _tmp79 * _w_pose_j[4]
    )
    _tmp125 = _K[0] * _tmp108 * _tmp109 * _tmp124 * _tmp52 - _tmp72 * (
        _tmp41 * (-_tmp102 - _tmp103 + _tmp107) + _tmp6 * (_tmp101 + _tmp97) + _tmp91
    )
    _tmp126 = -_tmp2
    _tmp127 = _tmp0 + _tmp113 + _tmp126
    _tmp128 = -_tmp121 - _tmp29
    _tmp129 = _K[1] * _tmp108 * _tmp109 * _tmp124 * _tmp83 - _tmp84 * (
        -_tmp127 * _w_pose_i[6]
        + _tmp127 * _w_pose_j[6]
        + _tmp36 * (_tmp102 + _tmp103 + _tmp107)
        + _tmp41 * (-_tmp117 - _tmp88)
        + _tmp6 * (-_tmp120 - _tmp128)
        + _tmp68
    )
    _tmp130 = -_tmp63
    _tmp131 = -_tmp127
    _tmp132 = _tmp57 * _tmp87
    _tmp133 = 2 * _tmp104
    _tmp134 = -_tmp22
    _tmp135 = _tmp134 * _tmp22
    _tmp136 = _tmp16 * _tmp92
    _tmp137 = _tmp11 * _tmp57
    _tmp138 = -_tmp137
    _tmp139 = _tmp37 * _tmp92
    _tmp140 = _tmp134 * _tmp31
    _tmp141 = (1.0 / 2.0) * _tmp10 + (1.0 / 2.0) * _tmp7 + (1.0 / 2.0) * _tmp8 + (1.0 / 2.0) * _tmp9
    _tmp142 = -_tmp141
    _tmp143 = _tmp142 * _tmp27
    _tmp144 = _tmp140 + _tmp143
    _tmp145 = _tmp111 + _tmp112 + _tmp126 + _tmp53
    _tmp146 = _tmp116 * _tmp134
    _tmp147 = -_tmp119 - _tmp134 * _tmp22
    _tmp148 = _tmp11 * _tmp134
    _tmp149 = _tmp27 * _tmp92
    _tmp150 = _tmp100 + _tmp96
    _tmp151 = (
        -_tmp145 * _w_pose_i[4]
        + _tmp145 * _w_pose_j[4]
        + _tmp36 * (-_tmp118 - _tmp136 - _tmp147)
        + _tmp41 * (_tmp148 + _tmp149 + _tmp150)
        + _tmp51
        + _tmp6 * (-_tmp133 - _tmp146)
    )
    _tmp152 = _K[0] * _tmp108 * _tmp109 * _tmp151 * _tmp52 - _tmp72 * (
        -_tmp130 * _w_pose_i[4]
        + _tmp130 * _w_pose_j[4]
        - _tmp131 * _w_pose_i[6]
        + _tmp131 * _w_pose_j[6]
        + _tmp36 * (-_tmp132 - _tmp133)
        + _tmp41 * (_tmp138 - _tmp139 + _tmp144)
        + _tmp6 * (_tmp120 + _tmp135 + _tmp136)
        - _tmp66 * _w_pose_i[5]
        + _tmp66 * _w_pose_j[5]
    )
    _tmp153 = _tmp41 * (-_tmp132 - _tmp146)
    _tmp154 = _K[1] * _tmp108 * _tmp109 * _tmp151 * _tmp83 - _tmp84 * (
        _tmp153 + _tmp36 * (_tmp137 + _tmp139 + _tmp144) + _tmp6 * (-_tmp148 - _tmp149 + _tmp150)
    )
    _tmp155 = -_tmp57
    _tmp156 = _tmp11 * _tmp155
    _tmp157 = _tmp105 * _tmp16
    _tmp158 = _tmp143 + _tmp89
    _tmp159 = 2 * _tmp155 * _tmp31
    _tmp160 = 2 * _tmp96
    _tmp161 = _tmp105 * _tmp37 + _tmp118
    _tmp162 = -_tmp99
    _tmp163 = _tmp162 * _tmp27
    _tmp164 = -_tmp163 - _tmp32
    _tmp165 = _K[0] * _tmp52
    _tmp166 = _tmp116 * _tmp31
    _tmp167 = _tmp6 * (-_tmp159 - _tmp166)
    _tmp168 = _tmp155 * _tmp22
    _tmp169 = _tmp142 * _tmp16 + _tmp168
    _tmp170 = _tmp11 * _tmp31
    _tmp171 = _tmp106 + _tmp170
    _tmp172 = -_tmp156
    _tmp173 = _tmp108 * _tmp109
    _tmp174 = _tmp173 * (
        _tmp167 + _tmp36 * (-_tmp157 + _tmp158 + _tmp172) + _tmp41 * (_tmp169 + _tmp171)
    )
    _tmp175 = _tmp165 * _tmp174 - _tmp72 * (
        -_tmp114 * _w_pose_i[5]
        + _tmp114 * _w_pose_j[5]
        + _tmp36 * (-_tmp159 - _tmp160)
        + _tmp41 * (-_tmp161 - _tmp164)
        + _tmp6 * (_tmp156 + _tmp157 + _tmp158)
        + _tmp82
    )
    _tmp176 = -_tmp45
    _tmp177 = -_tmp145
    _tmp178 = _tmp163 + _tmp32
    _tmp179 = -_tmp170
    _tmp180 = -_tmp106 + _tmp179
    _tmp181 = _K[1] * _tmp83
    _tmp182 = _tmp174 * _tmp181 - _tmp84 * (
        -_tmp176 * _w_pose_i[5]
        + _tmp176 * _w_pose_j[5]
        - _tmp177 * _w_pose_i[4]
        + _tmp177 * _w_pose_j[4]
        + _tmp36 * (_tmp161 + _tmp178)
        + _tmp41 * (-_tmp160 - _tmp166)
        - _tmp49 * _w_pose_i[6]
        + _tmp49 * _w_pose_j[6]
        + _tmp6 * (_tmp169 + _tmp180)
    )
    _tmp183 = _tmp130 * _tmp173
    _tmp184 = _tmp165 * _tmp183 - _tmp4 * _tmp72
    _tmp185 = _tmp181 * _tmp183 - _tmp79 * _tmp84
    _tmp186 = _tmp173 * _tmp66
    _tmp187 = _tmp165 * _tmp186 - _tmp176 * _tmp72
    _tmp188 = _tmp181 * _tmp186 - _tmp74 * _tmp84
    _tmp189 = _K[0] * _tmp108 * _tmp109 * _tmp52 * _tmp55 - _tmp49 * _tmp72
    _tmp190 = _K[1] * _tmp108 * _tmp109 * _tmp55 * _tmp83 - _tmp110 * _tmp84
    _tmp191 = 2 * _tmp98
    _tmp192 = _tmp16 * _tmp162
    _tmp193 = _tmp102 + _tmp162 * _tmp37
    _tmp194 = 2 * _tmp137
    _tmp195 = _tmp11 ** 2
    _tmp196 = _tmp163 + _tmp195
    _tmp197 = _tmp173 * (
        _tmp36 * (-_tmp192 + _tmp94) + _tmp41 * (_tmp122 + _tmp196) + _tmp6 * (-_tmp194 - _tmp90)
    )
    _tmp198 = _tmp165 * _tmp197 - _tmp72 * (
        _tmp41 * (-_tmp180 - _tmp193) + _tmp6 * (_tmp191 + _tmp192 + _tmp94) + _tmp91
    )
    _tmp199 = _tmp181 * _tmp197 - _tmp84 * (
        _tmp36 * (_tmp171 + _tmp193) + _tmp41 * (-_tmp194 - _tmp88) + _tmp6 * (-_tmp128 - _tmp196)
    )
    _tmp200 = 2 * _tmp170
    _tmp201 = _tmp121 + _tmp195
    _tmp202 = _tmp37 * _tmp93
    _tmp203 = _tmp141 * _tmp27
    _tmp204 = _tmp140 + _tmp203
    _tmp205 = _tmp148 + _tmp94
    _tmp206 = _tmp173 * (
        _tmp36 * (-_tmp147 - _tmp201) + _tmp41 * (_tmp101 + _tmp205) + _tmp6 * (-_tmp146 - _tmp200)
    )
    _tmp207 = _tmp165 * _tmp206 - _tmp72 * (
        _tmp36 * (-_tmp132 - _tmp200)
        + _tmp41 * (_tmp138 - _tmp202 + _tmp204)
        + _tmp6 * (_tmp119 + _tmp135 + _tmp201)
    )
    _tmp208 = _tmp181 * _tmp206 - _tmp84 * (
        _tmp153 + _tmp36 * (_tmp137 + _tmp202 + _tmp204) + _tmp6 * (-_tmp123 - _tmp205)
    )
    _tmp209 = -_tmp105
    _tmp210 = _tmp195 + _tmp209 * _tmp37
    _tmp211 = _tmp16 * _tmp209
    _tmp212 = _tmp203 + _tmp89
    _tmp213 = _tmp209 * _tmp27
    _tmp214 = _tmp141 * _tmp16 + _tmp168
    _tmp215 = _tmp173 * (
        _tmp167 + _tmp36 * (_tmp172 - _tmp211 + _tmp212) + _tmp41 * (_tmp170 + _tmp213 + _tmp214)
    )
    _tmp216 = _tmp165 * _tmp215 - _tmp72 * (
        _tmp36 * (-_tmp159 - _tmp191)
        + _tmp41 * (-_tmp164 - _tmp210)
        + _tmp6 * (_tmp156 + _tmp211 + _tmp212)
    )
    _tmp217 = _tmp181 * _tmp215 - _tmp84 * (
        _tmp36 * (_tmp178 + _tmp210)
        + _tmp41 * (-_tmp166 - _tmp191)
        + _tmp6 * (_tmp179 - _tmp213 + _tmp214)
    )
    _tmp218 = _tmp173 * _tmp63
    _tmp219 = _tmp165 * _tmp218 - _tmp5 * _tmp72
    _tmp220 = _tmp181 * _tmp218 - _tmp80 * _tmp84
    _tmp221 = _tmp173 * _tmp67
    _tmp222 = _tmp165 * _tmp221 - _tmp45 * _tmp72
    _tmp223 = _tmp181 * _tmp221 - _tmp75 * _tmp84
    _tmp224 = _K[0] * _tmp108 * _tmp109 * _tmp52 * _tmp56 - _tmp50 * _tmp72
    _tmp225 = _K[1] * _tmp108 * _tmp109 * _tmp56 * _tmp83 - _tmp81 * _tmp84
    _tmp226 = d_src ** (-2)
    _tmp227 = _tmp226 * _tmp35
    _tmp228 = _tmp226 * _tmp40
    _tmp229 = -_tmp226 * _tmp59 - _tmp227 * _tmp60 - _tmp228 * _tmp62
    _tmp230 = _K[0] * _tmp108 * _tmp109 * _tmp229 * _tmp52 - _tmp72 * (
        -_tmp226 * _tmp28 - _tmp227 * _tmp34 - _tmp228 * _tmp39
    )
    _tmp231 = _K[1] * _tmp108 * _tmp109 * _tmp229 * _tmp83 - _tmp84 * (
        -_tmp226 * _tmp76 - _tmp227 * _tmp77 - _tmp228 * _tmp78
    )

    # Output terms
    _res = numpy.zeros(2)
    _res[0] = _tmp73
    _res[1] = _tmp85
    _jacobian = numpy.zeros((2, 13))
    _jacobian[0, 0] = _tmp125
    _jacobian[1, 0] = _tmp129
    _jacobian[0, 1] = _tmp152
    _jacobian[1, 1] = _tmp154
    _jacobian[0, 2] = _tmp175
    _jacobian[1, 2] = _tmp182
    _jacobian[0, 3] = _tmp184
    _jacobian[1, 3] = _tmp185
    _jacobian[0, 4] = _tmp187
    _jacobian[1, 4] = _tmp188
    _jacobian[0, 5] = _tmp189
    _jacobian[1, 5] = _tmp190
    _jacobian[0, 6] = _tmp198
    _jacobian[1, 6] = _tmp199
    _jacobian[0, 7] = _tmp207
    _jacobian[1, 7] = _tmp208
    _jacobian[0, 8] = _tmp216
    _jacobian[1, 8] = _tmp217
    _jacobian[0, 9] = _tmp219
    _jacobian[1, 9] = _tmp220
    _jacobian[0, 10] = _tmp222
    _jacobian[1, 10] = _tmp223
    _jacobian[0, 11] = _tmp224
    _jacobian[1, 11] = _tmp225
    _jacobian[0, 12] = _tmp230
    _jacobian[1, 12] = _tmp231
    _hessian = numpy.zeros((13, 13))
    _hessian[0, 0] = _tmp125 ** 2 + _tmp129 ** 2
    _hessian[1, 0] = _tmp125 * _tmp152 + _tmp129 * _tmp154
    _hessian[2, 0] = _tmp125 * _tmp175 + _tmp129 * _tmp182
    _hessian[3, 0] = _tmp125 * _tmp184 + _tmp129 * _tmp185
    _hessian[4, 0] = _tmp125 * _tmp187 + _tmp129 * _tmp188
    _hessian[5, 0] = _tmp125 * _tmp189 + _tmp129 * _tmp190
    _hessian[6, 0] = _tmp125 * _tmp198 + _tmp129 * _tmp199
    _hessian[7, 0] = _tmp125 * _tmp207 + _tmp129 * _tmp208
    _hessian[8, 0] = _tmp125 * _tmp216 + _tmp129 * _tmp217
    _hessian[9, 0] = _tmp125 * _tmp219 + _tmp129 * _tmp220
    _hessian[10, 0] = _tmp125 * _tmp222 + _tmp129 * _tmp223
    _hessian[11, 0] = _tmp125 * _tmp224 + _tmp129 * _tmp225
    _hessian[12, 0] = _tmp125 * _tmp230 + _tmp129 * _tmp231
    _hessian[0, 1] = 0
    _hessian[1, 1] = _tmp152 ** 2 + _tmp154 ** 2
    _hessian[2, 1] = _tmp152 * _tmp175 + _tmp154 * _tmp182
    _hessian[3, 1] = _tmp152 * _tmp184 + _tmp154 * _tmp185
    _hessian[4, 1] = _tmp152 * _tmp187 + _tmp154 * _tmp188
    _hessian[5, 1] = _tmp152 * _tmp189 + _tmp154 * _tmp190
    _hessian[6, 1] = _tmp152 * _tmp198 + _tmp154 * _tmp199
    _hessian[7, 1] = _tmp152 * _tmp207 + _tmp154 * _tmp208
    _hessian[8, 1] = _tmp152 * _tmp216 + _tmp154 * _tmp217
    _hessian[9, 1] = _tmp152 * _tmp219 + _tmp154 * _tmp220
    _hessian[10, 1] = _tmp152 * _tmp222 + _tmp154 * _tmp223
    _hessian[11, 1] = _tmp152 * _tmp224 + _tmp154 * _tmp225
    _hessian[12, 1] = _tmp152 * _tmp230 + _tmp154 * _tmp231
    _hessian[0, 2] = 0
    _hessian[1, 2] = 0
    _hessian[2, 2] = _tmp175 ** 2 + _tmp182 ** 2
    _hessian[3, 2] = _tmp175 * _tmp184 + _tmp182 * _tmp185
    _hessian[4, 2] = _tmp175 * _tmp187 + _tmp182 * _tmp188
    _hessian[5, 2] = _tmp175 * _tmp189 + _tmp182 * _tmp190
    _hessian[6, 2] = _tmp175 * _tmp198 + _tmp182 * _tmp199
    _hessian[7, 2] = _tmp175 * _tmp207 + _tmp182 * _tmp208
    _hessian[8, 2] = _tmp175 * _tmp216 + _tmp182 * _tmp217
    _hessian[9, 2] = _tmp175 * _tmp219 + _tmp182 * _tmp220
    _hessian[10, 2] = _tmp175 * _tmp222 + _tmp182 * _tmp223
    _hessian[11, 2] = _tmp175 * _tmp224 + _tmp182 * _tmp225
    _hessian[12, 2] = _tmp175 * _tmp230 + _tmp182 * _tmp231
    _hessian[0, 3] = 0
    _hessian[1, 3] = 0
    _hessian[2, 3] = 0
    _hessian[3, 3] = _tmp184 ** 2 + _tmp185 ** 2
    _hessian[4, 3] = _tmp184 * _tmp187 + _tmp185 * _tmp188
    _hessian[5, 3] = _tmp184 * _tmp189 + _tmp185 * _tmp190
    _hessian[6, 3] = _tmp184 * _tmp198 + _tmp185 * _tmp199
    _hessian[7, 3] = _tmp184 * _tmp207 + _tmp185 * _tmp208
    _hessian[8, 3] = _tmp184 * _tmp216 + _tmp185 * _tmp217
    _hessian[9, 3] = _tmp184 * _tmp219 + _tmp185 * _tmp220
    _hessian[10, 3] = _tmp184 * _tmp222 + _tmp185 * _tmp223
    _hessian[11, 3] = _tmp184 * _tmp224 + _tmp185 * _tmp225
    _hessian[12, 3] = _tmp184 * _tmp230 + _tmp185 * _tmp231
    _hessian[0, 4] = 0
    _hessian[1, 4] = 0
    _hessian[2, 4] = 0
    _hessian[3, 4] = 0
    _hessian[4, 4] = _tmp187 ** 2 + _tmp188 ** 2
    _hessian[5, 4] = _tmp187 * _tmp189 + _tmp188 * _tmp190
    _hessian[6, 4] = _tmp187 * _tmp198 + _tmp188 * _tmp199
    _hessian[7, 4] = _tmp187 * _tmp207 + _tmp188 * _tmp208
    _hessian[8, 4] = _tmp187 * _tmp216 + _tmp188 * _tmp217
    _hessian[9, 4] = _tmp187 * _tmp219 + _tmp188 * _tmp220
    _hessian[10, 4] = _tmp187 * _tmp222 + _tmp188 * _tmp223
    _hessian[11, 4] = _tmp187 * _tmp224 + _tmp188 * _tmp225
    _hessian[12, 4] = _tmp187 * _tmp230 + _tmp188 * _tmp231
    _hessian[0, 5] = 0
    _hessian[1, 5] = 0
    _hessian[2, 5] = 0
    _hessian[3, 5] = 0
    _hessian[4, 5] = 0
    _hessian[5, 5] = _tmp189 ** 2 + _tmp190 ** 2
    _hessian[6, 5] = _tmp189 * _tmp198 + _tmp190 * _tmp199
    _hessian[7, 5] = _tmp189 * _tmp207 + _tmp190 * _tmp208
    _hessian[8, 5] = _tmp189 * _tmp216 + _tmp190 * _tmp217
    _hessian[9, 5] = _tmp189 * _tmp219 + _tmp190 * _tmp220
    _hessian[10, 5] = _tmp189 * _tmp222 + _tmp190 * _tmp223
    _hessian[11, 5] = _tmp189 * _tmp224 + _tmp190 * _tmp225
    _hessian[12, 5] = _tmp189 * _tmp230 + _tmp190 * _tmp231
    _hessian[0, 6] = 0
    _hessian[1, 6] = 0
    _hessian[2, 6] = 0
    _hessian[3, 6] = 0
    _hessian[4, 6] = 0
    _hessian[5, 6] = 0
    _hessian[6, 6] = _tmp198 ** 2 + _tmp199 ** 2
    _hessian[7, 6] = _tmp198 * _tmp207 + _tmp199 * _tmp208
    _hessian[8, 6] = _tmp198 * _tmp216 + _tmp199 * _tmp217
    _hessian[9, 6] = _tmp198 * _tmp219 + _tmp199 * _tmp220
    _hessian[10, 6] = _tmp198 * _tmp222 + _tmp199 * _tmp223
    _hessian[11, 6] = _tmp198 * _tmp224 + _tmp199 * _tmp225
    _hessian[12, 6] = _tmp198 * _tmp230 + _tmp199 * _tmp231
    _hessian[0, 7] = 0
    _hessian[1, 7] = 0
    _hessian[2, 7] = 0
    _hessian[3, 7] = 0
    _hessian[4, 7] = 0
    _hessian[5, 7] = 0
    _hessian[6, 7] = 0
    _hessian[7, 7] = _tmp207 ** 2 + _tmp208 ** 2
    _hessian[8, 7] = _tmp207 * _tmp216 + _tmp208 * _tmp217
    _hessian[9, 7] = _tmp207 * _tmp219 + _tmp208 * _tmp220
    _hessian[10, 7] = _tmp207 * _tmp222 + _tmp208 * _tmp223
    _hessian[11, 7] = _tmp207 * _tmp224 + _tmp208 * _tmp225
    _hessian[12, 7] = _tmp207 * _tmp230 + _tmp208 * _tmp231
    _hessian[0, 8] = 0
    _hessian[1, 8] = 0
    _hessian[2, 8] = 0
    _hessian[3, 8] = 0
    _hessian[4, 8] = 0
    _hessian[5, 8] = 0
    _hessian[6, 8] = 0
    _hessian[7, 8] = 0
    _hessian[8, 8] = _tmp216 ** 2 + _tmp217 ** 2
    _hessian[9, 8] = _tmp216 * _tmp219 + _tmp217 * _tmp220
    _hessian[10, 8] = _tmp216 * _tmp222 + _tmp217 * _tmp223
    _hessian[11, 8] = _tmp216 * _tmp224 + _tmp217 * _tmp225
    _hessian[12, 8] = _tmp216 * _tmp230 + _tmp217 * _tmp231
    _hessian[0, 9] = 0
    _hessian[1, 9] = 0
    _hessian[2, 9] = 0
    _hessian[3, 9] = 0
    _hessian[4, 9] = 0
    _hessian[5, 9] = 0
    _hessian[6, 9] = 0
    _hessian[7, 9] = 0
    _hessian[8, 9] = 0
    _hessian[9, 9] = _tmp219 ** 2 + _tmp220 ** 2
    _hessian[10, 9] = _tmp219 * _tmp222 + _tmp220 * _tmp223
    _hessian[11, 9] = _tmp219 * _tmp224 + _tmp220 * _tmp225
    _hessian[12, 9] = _tmp219 * _tmp230 + _tmp220 * _tmp231
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
    _hessian[10, 10] = _tmp222 ** 2 + _tmp223 ** 2
    _hessian[11, 10] = _tmp222 * _tmp224 + _tmp223 * _tmp225
    _hessian[12, 10] = _tmp222 * _tmp230 + _tmp223 * _tmp231
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
    _hessian[11, 11] = _tmp224 ** 2 + _tmp225 ** 2
    _hessian[12, 11] = _tmp224 * _tmp230 + _tmp225 * _tmp231
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
    _hessian[12, 12] = _tmp230 ** 2 + _tmp231 ** 2
    _rhs = numpy.zeros(13)
    _rhs[0] = _tmp125 * _tmp73 + _tmp129 * _tmp85
    _rhs[1] = _tmp152 * _tmp73 + _tmp154 * _tmp85
    _rhs[2] = _tmp175 * _tmp73 + _tmp182 * _tmp85
    _rhs[3] = _tmp184 * _tmp73 + _tmp185 * _tmp85
    _rhs[4] = _tmp187 * _tmp73 + _tmp188 * _tmp85
    _rhs[5] = _tmp189 * _tmp73 + _tmp190 * _tmp85
    _rhs[6] = _tmp198 * _tmp73 + _tmp199 * _tmp85
    _rhs[7] = _tmp207 * _tmp73 + _tmp208 * _tmp85
    _rhs[8] = _tmp216 * _tmp73 + _tmp217 * _tmp85
    _rhs[9] = _tmp219 * _tmp73 + _tmp220 * _tmp85
    _rhs[10] = _tmp222 * _tmp73 + _tmp223 * _tmp85
    _rhs[11] = _tmp224 * _tmp73 + _tmp225 * _tmp85
    _rhs[12] = _tmp230 * _tmp73 + _tmp231 * _tmp85
    return _res, _jacobian, _hessian, _rhs
