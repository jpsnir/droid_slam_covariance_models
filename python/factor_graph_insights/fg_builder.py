"""
This module provides utilities for creating a factor graph
from a given .pkl file obtained from droid slam.
The factor graph is constructed using gtsam.
The custom factor generated from symforce need to be integrated with proper
datatype dependency relation between gtsam and sym module.
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

init_values = gtsam.Values()


class DataConverter:
    @staticmethod
    def to_gtsam_pose(pose: Union[np.ndarray, torch.Tensor]) -> gtsam.Pose3:
        """
        converts nd array pose to gtsam.Pose3
        pose in nd array is tx, ty, tz, qw, qx, qy, qz
        """
        if isinstance(pose, torch.Tensor):
            pose = pose.numpy()
        assert pose.shape == (7,), "Pose is not 7x1 numpy array"
        translation = pose[:3]
        rotation = gtsam.Rot3(w=pose[6], x=pose[3], y=pose[4], z=pose[5])
        return gtsam.Pose3(r=rotation, t=translation)


class ImagePairFactorGraphBuilder:
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
        assert (self._K is not None, "Calibration parameters are not set")
        return torch.tensor(self._K)

    @property
    def camera(self) -> gtsam.Cal3_S2:
        assert (self._cal3s2_camera is not None, "Calibration parameters are not set")
        return self._cal3s2_camera

    def set_calibration(self, calibration: torch.Tensor) -> Self:
        self._K = calibration.numpy()
        self._cal3s2_camera = gtsam.Cal3_S2(
            self._K[0],
            self._K[1],
            0,
            self._K[2],
            self._K[3],
        )
        return self

    @property
    def poses(self):
        assert self._pose_i is not None, "Pose i does not exist"
        assert self._pose_j is not None, "Pose j does not exist"

        return self._pose_i, self._pose_j

    def set_poses(self, pose_i: torch.Tensor, pose_j: torch.Tensor) -> Self:
        """ """

        self._pose_i = pose_i
        self._pose_j = pose_j
        return self

    @property
    def pinhole_cameras(
        self,
    ) -> Tuple[gtsam.PinholeCameraCal3_S2, gtsam.PinholeCameraCal3_S2]:
        """get pinhole camera objects"""
        assert self._ph_camera_i is not None, "pinhole camera i is not created"
        assert self._ph_camera_j is not None, "pinhole camera j is not created"

        return self._ph_camera_i, self._ph_camera_j

    def create_pinhole_cameras(self) -> Self:
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
            target_pts.shape == self._image_size,
            " Target point tensor does not match image size",
        )
        self._target_pts = target_pts
        return self

    @property
    def depths(self):
        assert self._depths is not None, " Depths are not set."
        return self._depths

    def set_depths(self, depths: torch.Tensor) -> Self:
        """"""
        assert (
            depths.shape == self._image_size,
            " Target depth size does not match image size",
        )
        self._depths = depths
        return self

    @property
    def pixel_weights(self) -> torch.Tensor:
        assert self._weights is not None, "Pixel weights are not set"
        return self._weights

    def set_pixel_weights(self, weights: torch.Tensor) -> Self:
        assert (
            weights.shape == self._image_size,
            f" Target weight size does not match image size",
        )
        self._weights = weights
        return self

    @property
    def error_model(self) -> gtsam.CustomFactor:
        assert self._error_model is not None, "Custom factor is not set"
        return self._error_model

    @error_model.setter
    def error_model(self, error_model: object):
        """assigns the custom factor"""

        assert (
            getattr(error_model, "error"),
            "No attribute error function in error model",
        )
        assert (callable(error_model.error), "error attribute is not callable")
        self._error_model = error_model
        return self

    def is_pt_close_to_cam(
        self, pixel_i: Union[Tuple[int, int], np.ndarray], min_depth: float
    ) -> bool:
        if isinstance(pixel_i, Tuple):
            assert (
                len(pixel_i) == 2
            ), "Shape mismatch, - row, columns - two values required"
            pixel_i = np.array(pixel_i)
        if isinstance(pixel_i, np.ndarray):
            assert pixel_i.shape == (2,), "Shape mismatch - required (2,)"

        row, col = pixel_i
        depth_i = self._depths[row, col].item()
        pt3d_w = self._ph_camera_i.backproject(pixel_i, depth_i)
        gtsam_pose_j = DataConverter.to_gtsam_pose(self._pose_j)
        pt3d_j = gtsam_pose_j.inverse().transformTo(pt3d_w)
        depth_j = pt3d_j[2]
        return depth_j < min_depth

    def build_visual_factor_graph(self) -> gtsam.NonlinearFactorGraph:
        """
        build a non-linear factor graph
        """
        ROWS, COLS = self._image_size
        s_x_i = gtsam.symbol("x", self.i)
        s_x_j = gtsam.symbol("x", self.i)
        for row in range(ROWS):
            for col in range(COLS):
                # each depth in ith camera has to be assigned a symbol
                # as it will be optimized as a variable.
                depth_flag = self.is_pt_close_to_cam((row, col), 0.25)

                s_d_i = gtsam.symbol("d", ROWS * COLS * self.i + count_symbol)
                if not init_values.exists(symbol_di):
                    init_values.insert(symbol_di, depth[row, col].numpy())
                # define noise for each pixel from confidence map or weight
                # matrix
                print(f"depth of point in {j} camera - {depth_j} - {depth_j < 0.25}")
                if depth_j < 0.25:
                    w = np.array([0, 0])
                else:
                    w = 0.001 * weights[:, row, col].numpy().reshape(2)
                self._error_model.make_custom_factor()
                graph.add(self._error_model.custom_factor)
                count_symbol += 1


def factor_graph_image_pair(
    i: int,
    j: int,
    pair_id: int,
    pose_i: Union[np.ndarray, torch.Tensor],
    pose_j: Union[np.ndarray, torch.Tensor],
    depth: Union[np.ndarray, torch.Tensor],
    weights: Union[np.ndarray, torch.Tensor],
    target_pt: Union[np.ndarray, torch.Tensor],
    intrinsics: Union[np.ndarray, torch.Tensor],
) -> gtsam.NonlinearFactorGraph:
    """
    generates the factor graph of one image pair

    :param weights: weights(i->j) assigned for each pixel in the given each image pair
    """
    # todo check sizes of each matrix
    ROWS = depth.size()[0]
    COLS = depth.size()[1]
    symbol_xi = gtsam.symbol("x", i)
    symbol_xj = gtsam.symbol("x", j)
    if isinstance(intrinsics, torch.Tensor):
        k_matrix = intrinsics.numpy()
    else:
        k_matrix = intrinsics
    pose_i_gtsam = convert_to_gtsam_pose(pose_i.numpy())
    pose_j_gtsam = convert_to_gtsam_pose(pose_j.numpy())
    if not init_values.exists(symbol_xi):
        init_values.insert(symbol_xi, pose_i_gtsam)
    if not init_values.exists(symbol_xj):
        init_values.insert(symbol_xj, pose_j_gtsam)

    # define pinhole model
    camera = gtsam.Cal3_S2(k_matrix[0], k_matrix[1], 0, k_matrix[2], k_matrix[3])
    # perspective camera model
    ph_camera_i = gtsam.PinholePoseCal3_S2(
        pose=pose_i_gtsam,
        K=camera,
    )
    ph_camera_j = gtsam.PinholePoseCal3_S2(
        pose=pose_j_gtsam,
        K=camera,
    )
    graph = gtsam.NonlinearFactorGraph()
    # Poses are to be optimized.
    # X_i and X_j so they are assigned symbols.
    count_symbol = 0
    for row, d_row in enumerate(depth):
        for col, d in enumerate(d_row):
            # each depth in ith camera has to be assigned a symbol
            # as it will be optimized as a variable.

            symbol_di = gtsam.symbol("d", ROWS * COLS * i + count_symbol)

            if not init_values.exists(symbol_di):
                init_values.insert(symbol_di, depth[row, col].numpy())
            # define noise for each pixel from confidence map or weight
            # matrix
            xy_i = np.array([row, col])
            depth_xy_i = depth[row, col].item()
            pt3d_w = ph_camera_i.backproject(xy_i, depth_xy_i)
            pt3d_j = pose_j_gtsam.inverse().transformTo(pt3d_w)
            depth_j = pt3d_j[2]
            print(f"depth of point in {j} camera - {depth_j} - {depth_j < 0.25}")
            if depth_j < 0.25:
                w = np.array([0, 0])
            else:
                w = 0.001 * weights[:, row, col].numpy().reshape(2)
            pixel_noise_model = gtsam.noiseModel.Diagonal.Information(np.diag(w))
            pixel_noise_model.print()
            dst_img_coords = target_pt[:, row, col].numpy().reshape(2, 1)
            src_img_coords = np.array([row, col]).reshape(2, 1)
            # define factor for the pixel at (row, col)
            keys = gtsam.KeyVector([symbol_xi, symbol_xj, symbol_di])
            custom_factor = gtsam.CustomFactor(
                pixel_noise_model,
                keys,
                partial(
                    droid_slam_error_func,
                    dst_img_coords,
                    src_img_coords,
                    intrinsics,
                ),
            )
            graph.add(custom_factor)
            count_symbol += 1
    return graph


def build_factor_graph(fg_data: dict, n: int = 0) -> gtsam.NonlinearFactorGraph:
    """
    build factor graph from complete data
    """
    graph_data = fg_data["graph_data"]
    depth = fg_data["disps"]
    poses = fg_data["poses"]
    weights = fg_data["c_map"]
    predicted = fg_data["predicted"]
    K = fg_data["intrinsics"]
    ii = graph_data["ii"]
    jj = graph_data["jj"]
    pair_unique_id = {}
    if n == 0:
        n = ii.size()[0]
    unique_id = 0
    full_graph = gtsam.NonlinearFactorGraph()
    print(
        f"""Graph index - {graph_data['ii'].size()},
              poses - {poses.size()},
              ---------------------------
              weights - shape = {weights.size()},
              ---------------------------
              predicted - shape = {predicted.size()},
              ---------------------------
              depth - shape = {depth.size()},
              -------------------------------
              intrinics - shape = {K.size()}, {K},
        """
    )
    for index, (ix, jx) in enumerate(zip(ii[:n], jj[:n])):
        key = (ix, jx)
        if key not in pair_unique_id.keys():
            pair_unique_id[key] = unique_id
            unique_id += 1
        if max(ix, jx).item() > poses.size()[0] - 1:
            print(f"Ignoring index - {ix , jx} - out of bounds")
            continue
        print(f"Index - {index} - Adding factors for {ix} - {jx} edge")
        graph = factor_graph_image_pair(
            i=ix,
            j=jx,
            pair_id=unique_id,
            pose_i=poses[ix],
            pose_j=poses[jx],
            depth=depth[ix],
            weights=weights[index],
            target_pt=predicted[index],
            intrinsics=K,
        )
        full_graph.push_back(graph)
    print(f"Number of factors in full factor graph = {full_graph.nrFactors()}")
    return full_graph


if __name__ == "__main__":
    N = 5
    fg_dir = Path("/media/jagatpreet/D/datasets/uw_rig/samples").joinpath(
        "woodshole_east_dock_1/factorgraph_data_2023_11_27_16_10_29"
    )
    if fg_dir.exists():
        files_list = sorted(os.listdir(fg_dir))
    print(f"Number of files = {len(files_list)}")
    fg_file = fg_dir.joinpath(files_list[0])

    # Prior noise definition for first two poses.
    #  3D rotational standard deviation of prior factor - gaussian model
    #  (degrees)
    prior_rpy_sigma = 1
    # 3D translational standard deviation of of prior factor - gaussian model
    # (meters)
    prior_xyz_sigma = 0.05
    sigma_angle = np.deg2rad(prior_rpy_sigma)
    prior_noise_model = gtsam.noiseModel.Diagonal.Sigmas(
        np.array(
            [
                sigma_angle,
                sigma_angle,
                sigma_angle,
                prior_xyz_sigma,
                prior_xyz_sigma,
                prior_xyz_sigma,
            ]
        )
    )
    symbol_first_pose = gtsam.symbol("x", 0)
    symbol_second_pose = gtsam.symbol("x", 1)
    print(f"Analyzing file : { fg_file}")
    fg_data = import_fg_from_pickle_file(fg_file)
    print_factor_graph_stats(fg_data)
    graph = build_factor_graph(fg_data, N)
    graph.push_back(
        gtsam.PriorFactorPose3(
            symbol_first_pose, init_values.atPose3(symbol_first_pose), prior_noise_model
        )
    )
    graph.push_back(
        gtsam.PriorFactorPose3(
            symbol_second_pose,
            init_values.atPose3(symbol_first_pose),
            prior_noise_model,
        )
    )

    print(f"Number of factors={graph.nrFactors()}")
    flag = input("Linearize graph initial values: 0-> No, 1-> yes")
    jac_list = []
    b_list = []
    cov_list = []
    info_list = []
    if int(flag):
        print(f"Errors init values = {graph.error(init_values)}")
        lin_graph1 = graph.linearize(init_values)
        jac, b = lin_graph1.jacobian()
        cov = np.linalg.inv(jac.transpose() @ jac)
        info = jac.transpose() @ jac
    jac_list.append(jac)
    b_list.append(b)
    info_list.append(info)
    cov_list.append(cov)
    jac_list.append(jac)
    b_list.append(b)
    info_list.append(info)
    cov_list.append(cov)
    marginals_init = gtsam.Marginals(graph, init_values)
    sys.exit(0)
    number_of_iters = input("Enter integer number of iterations for optimization:")
    print(f"Number of iterations {number_of_iters}")
    time.sleep(2)
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(int(number_of_iters))
    print(f" LM params: {params}")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, init_values, params)
    result = optimizer.optimize()
    print(f"Final result :\n {result}")
    marginals_new = gtsam.Marginals(graph, result)
    flag = input("Linearize graph final values: 0-> No, 1-> yes")
    if int(flag):
        print(f"Errors final values = {graph.error(result)}")
        lin_graph2 = graph.linearize(result)
        jac, b = lin_graph2.jacobian()
        cov = np.linalg.inv(jac.transpose() @ jac)
        info = jac.transpose() @ jac
