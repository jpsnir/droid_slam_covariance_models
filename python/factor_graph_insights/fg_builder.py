"""
This module provides utilities for creating a factor graph of 
ternary factors for droid slam error function.
Creates a factor graph for a pair of images. 
@author : jagatpreet
"""
import os
import sys
from pathlib import Path
from functools import partial
import typing as T
from typing import (
    Union,
    List,
    Tuple,
    Dict,
    Optional,
)
from typing_extensions import Self
import numpy as np
import gtsam
import torch
from gtsam.symbol_shorthand import L, X

import factor_graph_insights.custom_factors as droid_autogen
from factor_graph_insights.custom_factors.droid_error_functions import Droid_DBA_Error
import time

# confidence map values will go here.

NEAR_DEPTH_THRESHOLD = 0.25


class DataConverter:
    @staticmethod
    def to_gtsam_pose(pose: Union[np.ndarray, torch.Tensor]) -> gtsam.Pose3:
        """
        converts nd array pose to gtsam.Pose3
        pose in nd array is tx, ty, tz, qw, qx, qy, qz
        """
        if isinstance(pose, torch.Tensor):
            pose = pose.numpy()
        assert pose.shape == (7,), "Pose is not 7x1 numpy array, size = {pose.shape}"
        translation = pose[:3]
        rotation = gtsam.Rot3(w=pose[6], x=pose[3], y=pose[4], z=pose[5])
        return gtsam.Pose3(r=rotation, t=translation)

    @staticmethod
    def to_pose_with_quaternion(gtsam_pose: gtsam.Pose3) -> np.ndarray:
        pose = np.zeros(7)
        t = gtsam_pose.translation()
        q = gtsam_pose.rotation().toQuaternion()
        pose[0] = t[0]
        pose[1] = t[1]
        pose[2] = t[2]
        pose[3] = q.x()
        pose[4] = q.y()
        pose[5] = q.z()
        pose[6] = q.w()
        return pose

    @staticmethod
    def invert_pose(pose: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        if isinstance(pose, torch.Tensor):
            pose = pose.numpy()
        inv_pose = torch.zeros(7)
        gtsam_pose = DataConverter.to_gtsam_pose(pose)
        gtsam_inv_pose = gtsam_pose.inverse()
        inv_pose = DataConverter.to_pose_with_quaternion(gtsam_inv_pose)
        return inv_pose


class FactorGraphBuilder:
    def __init__():
        """Stub initialization"""

    def build_factor_graph(self):
        """"""
        # stub method


class ImagePairFactorGraphBuilder(FactorGraphBuilder):
    """Image pair factor builder for an arbitrary image size"""

    def __init__(self, i, j, image_size: Tuple[int, int]) -> Self:
        """factor graph"""
        self.i = i
        self.j = j
        self._image_size = torch.Size(image_size)
        self._weights = None
        self._target_pts = None
        self._depths = None
        self._K = None
        self._cal3s2_camera = None
        self._custom_factor_residual_func = None
        self._ph_camera_i = None
        self._ph_camera_j = None
        self._custom_factor = None
        self._pose_i = None
        self._pose_j = None
        self._gtsam_pose_i = None
        self._gtsam_pose_j = None
        self._graph = None
        self._error_model = None
        self._init_values = gtsam.Values()

    @property
    def src_img_id(self) -> int:
        return self.i

    @src_img_id.setter
    def src_img_id(self, i: int):
        self.i = i

    @property
    def dst_img_id(self) -> int:
        return self.j

    @dst_img_id.setter
    def dst_img_id(self, j: int):
        self.j = j

    @property
    def image_size(self) -> Tuple[int, int]:
        return self._image_size

    @image_size.setter
    def image_size(self, image_size: Tuple[int, int]):
        self._image_size = torch.Size(image_size)

    @property
    def calibration(self) -> torch.Tensor:
        assert self._K is not None, "Calibration parameters are not set"
        return self._K

    @property
    def camera(self) -> gtsam.Cal3_S2:
        assert self._cal3s2_camera is not None, "Calibration parameters are not set"
        return self._cal3s2_camera

    def set_calibration(self, calibration: Union[torch.Tensor, np.ndarray]) -> Self:
        if isinstance(calibration, torch.Tensor):
            self._K = calibration.numpy()
        self._K = calibration
        self._cal3s2_camera = gtsam.Cal3_S2(
            self._K[0],
            self._K[1],
            self._K[2],
            self._K[3],
            self._K[4],
        )
        return self

    @property
    def poses(self):
        assert self._pose_i is not None, "Pose i does not exist"
        assert self._pose_j is not None, "Pose j does not exist"

        return self._pose_i, self._pose_j

    def set_poses_and_cameras(
        self,
        pose_i: Union[torch.Tensor, np.ndarray],
        pose_j: Union[torch.Tensor, np.ndarray],
    ) -> Self:
        """ """
        if isinstance(pose_i, np.ndarray):
            pose_i = torch.tensor(pose_i)
        if isinstance(pose_j, np.ndarray):
            pose_j = torch.tensor(pose_j)

        self._pose_i = pose_i
        self._pose_j = pose_j
        self._gtsam_pose_i = DataConverter.to_gtsam_pose(self._pose_i)
        self._gtsam_pose_j = DataConverter.to_gtsam_pose(self._pose_j)
        # as soon as you set new poses, pinhole cameras are created.
        self._create_pinhole_cameras()

        return self

    @property
    def pinhole_cameras(
        self,
    ) -> Tuple[gtsam.PinholeCameraCal3_S2, gtsam.PinholeCameraCal3_S2]:
        """get pinhole camera objects"""
        assert self._ph_camera_i is not None, "pinhole camera i is not created"
        assert self._ph_camera_j is not None, "pinhole camera j is not created"

        return self._ph_camera_i, self._ph_camera_j

    def _create_pinhole_cameras(self) -> Self:
        """make pinhole camera objects from poses and intrinsics"""
        assert (
            self._cal3s2_camera is not None
        ), "set cal2s2 camera object for  creating pinhole cameras"

        assert self._pose_i is not None, " Pose i is not set"
        assert self._pose_j is not None, " Pose j is not set"
        self._ph_camera_i = gtsam.PinholeCameraCal3_S2(
            pose=DataConverter.to_gtsam_pose(self._pose_i),
            K=self._cal3s2_camera,
        )

        self._ph_camera_j = gtsam.PinholeCameraCal3_S2(
            pose=DataConverter.to_gtsam_pose(self._pose_j),
            K=self._cal3s2_camera,
        )
        return self

    @property
    def target_pts(self):
        """"""
        return self._target_pts

    def set_target_pts(self, target_pts: torch.Tensor) -> Self:
        """"""
        assert (
            target_pts.shape[1],
            target_pts.shape[2],
        ) == self._image_size, " Target point tensor does not match image size"

        self._target_pts = target_pts
        return self

    @property
    def depths(self) -> torch.Tensor:
        assert self._depths is not None, " Depths are not set."
        return self._depths

    def depthAt(self, pixel: Union[Tuple, List]) -> float:
        assert self._depths is not None, "Depths are not set."
        assert len(pixel) == 2, "Input pixel should be a a Tuple or list of length 2"
        row, col = pixel
        return float(self._depths[row, col])

    def set_depths(self, depths: torch.Tensor) -> Self:
        """"""
        assert (
            depths.shape == self._image_size
        ), " Target depth size does not match image size"
        self._depths = depths
        return self

    @property
    def pixel_weights(self) -> torch.Tensor:
        assert self._weights is not None, "Pixel weights are not set"
        return self._weights

    def set_pixel_weights(self, weights: torch.Tensor) -> Self:
        assert (
            weights.shape[1],
            weights.shape[2],
        ) == self._image_size, f" Target weight {weights.shape} size does not match image size{self._image_size}"
        self._weights = weights
        return self

    @property
    def error_model(self) -> gtsam.CustomFactor:
        assert self._error_model is not None, "Custom factor is not set"
        return self._error_model

    @error_model.setter
    def error_model(self, error_model: object):
        """assigns the custom factor"""

        assert getattr(
            error_model, "error"
        ), "No attribute error function in error model"
        assert callable(error_model.error), "error attribute is not callable"
        self._error_model = error_model

    def set_error_model(self, error_model: object) -> Self:
        self.error_model = error_model
        return self

    @property
    def init_values_image_pair(self) -> gtsam.Values:
        return self._init_values

    @property
    def factor_graph(self) -> gtsam.NonlinearFactorGraph:
        assert self._graph is not None, "Factor graph is not defined"
        return self._graph

    def depth_to_cam_j(
        self,
        pixel_i: Union[Tuple[int, int], np.ndarray],
        depth_i,
        near_depth_threshold: float = 0.25,
    ) -> bool:
        if isinstance(pixel_i, Tuple):
            assert (
                len(pixel_i) == 2
            ), "Shape mismatch, - row, columns - two values required"
            pixel_i = np.array(pixel_i)
        if isinstance(pixel_i, np.ndarray):
            assert pixel_i.shape == (2,), "Shape mismatch - required (2,)"

        row, col = pixel_i
        # point in world
        pt3d_w = self._ph_camera_i.backproject(pixel_i, depth_i)
        # convert point to camera j coordinate system from world
        pt3d_j = self._gtsam_pose_j.transformTo(pt3d_w)
        depth_j = pt3d_j[2]
        is_near_j = depth_j < near_depth_threshold
        is_near_i = depth_i < near_depth_threshold
        return depth_j, (is_near_i, is_near_j)

    def _set_init_poses(self, symbols: Tuple[int, int]):
        if not self._init_values.exists(symbols[0]):
            self._init_values.insert(symbols[0], self._gtsam_pose_i)

        if not self._init_values.exists(symbols[1]):
            self._init_values.insert(symbols[1], self._gtsam_pose_j)

    def _set_init_depth(self, symbol, depth):
        if not self._init_values.exists(symbol):
            self._init_values.insert(symbol, depth)

    def build_factor_graph(
        self, cur_init_vals=gtsam.Values(), confidence_factor=0.001
    ) -> gtsam.NonlinearFactorGraph:
        """
        overrides base class
        build a non-linear factor graph
        """
        graph = gtsam.NonlinearFactorGraph()
        self._init_values = cur_init_vals

        ROWS, COLS = self._image_size
        s_x_i = gtsam.symbol("x", self.i)
        s_x_j = gtsam.symbol("x", self.j)
        self._set_init_poses((s_x_i, s_x_j))
        count_symbol = 0
        for row in range(ROWS):
            for col in range(COLS):
                # each depth in ith camera has to be assigned a symbol
                # as it will be optimized as a variable.
                depth_j, (is_close_to_cam_i, is_close_to_cam_j) = self.depth_to_cam_j(
                    (row, col), NEAR_DEPTH_THRESHOLD
                )

                if not (is_close_to_cam_i or is_close_to_cam_j):
                    s_d_i = gtsam.symbol("d", ROWS * COLS * self.i + count_symbol)
                    self._symbols = (s_x_i, s_x_j, s_d_i)
                    self._set_init_depth(s_d_i, depth_j)
                    pixel_confidence = (
                        confidence_factor * self._weights[:, row, col].numpy()
                    )
                    ## Add factor
                    assert pixel_confidence.shape == (2,)
                    pixel_to_project = np.array([row, col])
                    predicted_pixel = self._target_pts[:, row, col].numpy()
                    pixels = (pixel_to_project, predicted_pixel)
                    vars = (
                        self._gtsam_pose_i,
                        self._gtsam_pose_j,
                        self._depths[row, col],
                    )
                    self.error_model.make_custom_factor(
                        self._symbols,
                        pixels,
                        pixel_confidence,
                    )
                    graph.add(self._error_model.custom_factor)
                    count_symbol += 1
        return graph


# if __name__ == "__main__":
#     N = 5
#     fg_dir = Path("/media/jagatpreet/D/datasets/uw_rig/samples").joinpath(
#         "woodshole_east_dock_1/factorgraph_data_2023_11_27_16_10_29"
#     )
#     if fg_dir.exists():
#         files_list = sorted(os.listdir(fg_dir))
#     print(f"Number of files = {len(files_list)}")
#     fg_file = fg_dir.joinpath(files_list[0])

#     # Prior noise definition for first two poses.
#     #  3D rotational standard deviation of prior factor - gaussian model
#     #  (degrees)
#     prior_rpy_sigma = 1
#     # 3D translational standard deviation of of prior factor - gaussian model
#     # (meters)
#     prior_xyz_sigma = 0.05
#     sigma_angle = np.deg2rad(prior_rpy_sigma)
#     prior_noise_model = gtsam.noiseModel.Diagonal.Sigmas(
#         np.array(
#             [
#                 sigma_angle,
#                 sigma_angle,
#                 sigma_angle,
#                 prior_xyz_sigma,
#                 prior_xyz_sigma,
#                 prior_xyz_sigma,
#             ]
#         )
#     )
#     symbol_first_pose = gtsam.symbol("x", 0)
#     symbol_second_pose = gtsam.symbol("x", 1)
#     print(f"Analyzing file : { fg_file}")
#     fg_data = import_fg_from_pickle_file(fg_file)
#     print_factor_graph_stats(fg_data)
#     graph = build_factor_graph(fg_data, N)
#     graph.push_back(
#         gtsam.PriorFactorPose3(
#             symbol_first_pose, init_values.atPose3(symbol_first_pose), prior_noise_model
#         )
#     )
#     graph.push_back(
#         gtsam.PriorFactorPose3(
#             symbol_second_pose,
#             init_values.atPose3(symbol_first_pose),
#             prior_noise_model,
#         )
#     )

#     print(f"Number of factors={graph.nrFactors()}")
#     flag = input("Linearize graph initial values: 0-> No, 1-> yes")
#     jac_list = []
#     b_list = []
#     cov_list = []
#     info_list = []
#     if int(flag):
#         print(f"Errors init values = {graph.error(init_values)}")
#         lin_graph1 = graph.linearize(init_values)
#         jac, b = lin_graph1.jacobian()
#         cov = np.linalg.inv(jac.transpose() @ jac)
#         info = jac.transpose() @ jac
#     jac_list.append(jac)
#     b_list.append(b)
#     info_list.append(info)
#     cov_list.append(cov)
#     jac_list.append(jac)
#     b_list.append(b)
#     info_list.append(info)
#     cov_list.append(cov)
#     marginals_init = gtsam.Marginals(graph, init_values)
#     sys.exit(0)
#     number_of_iters = input("Enter integer number of iterations for optimization:")
#     print(f"Number of iterations {number_of_iters}")
#     time.sleep(2)
#     params = gtsam.LevenbergMarquardtParams()
#     params.setMaxIterations(int(number_of_iters))
#     print(f" LM params: {params}")
#     optimizer = gtsam.LevenbergMarquardtOptimizer(graph, init_values, params)
#     result = optimizer.optimize()
#     print(f"Final result :\n {result}")
#     marginals_new = gtsam.Marginals(graph, result)
#     flag = input("Linearize graph final values: 0-> No, 1-> yes")
#     if int(flag):
#         print(f"Errors final values = {graph.error(result)}")
#         lin_graph2 = graph.linearize(result)
#         jac, b = lin_graph2.jacobian()
#         cov = np.linalg.inv(jac.transpose() @ jac)
#         info = jac.transpose() @ jac
