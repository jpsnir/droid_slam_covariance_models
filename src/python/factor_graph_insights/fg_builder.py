'''
This module provides utilities for creating a factor graph
from a given .pkl file obtained from droid slam.
The factor graph is constructed using gtsam.
The custom factor generated from symforce need to be integrated with proper
datatype dependency relation between gtsam and sym module.
@author : jagatpreet
'''
import os
import gtsam
import sym
import pickle
import typing as T
from typing import (
    Union,
    List,
    Tuple,
    Dict,
    Optional,
)
import numpy as np
import torch
from pathlib import Path
from gtsam.symbol_shorthand import L, X
from functools import partial
import factor_graph_insights.custom_factors as droid_autogen
import time

# confidence map values will go here.

init_values = gtsam.Values()


def import_fg_from_pickle_file(filename: Union[str, Path]) -> Dict:
    '''
    generates parsed data from pickle file
    '''
    fg_data = {}
    with open(filename, 'rb') as f:
        fg_data = pickle.load(f)

    return fg_data


def print_factor_graph_stats(fg_data: dict):
    '''
    prints some basic details of the factor graph for
    error checking
    '''
    print(f'data keys: {fg_data.keys()}')
    print(f"id: {fg_data['id']}")
    print(f"intrinsics: {fg_data['intrinsics']}")
    print("------------------------------")
    print(f"graph data - ii: {fg_data['graph_data']['ii']}")
    print(f"graph data - jj: {fg_data['graph_data']['jj']}")
    print(f"graph data - number of connections:\
          {fg_data['graph_data']['ii'].size().numel()}")
    print(f"weights size: {fg_data['c_map'].size()}")
    print(f"target pts size: {fg_data['predicted'].size()}")
    print("------------------------------")
    print(f"tstamps size: {fg_data['tstamp'].size()}")
    print(f"poses size: {fg_data['poses'].size()}")
    print(f"disparity size: {fg_data['disps'].size()}")


def convert_to_gtsam_pose(pose: np.ndarray) -> gtsam.Pose3:
    '''
    converts nd array pose to gtsam.Pose3
    pose in nd array is tx, ty, tz, qw, qx, qy, qz
    '''

    translation = pose[:3]
    rotation = gtsam.Rot3(w=pose[6], x=pose[3], y=pose[4], z=pose[5])
    return gtsam.Pose3(r=rotation, t=translation)


def gtsam2sym_pose(pose: gtsam.Pose3) -> sym.Pose3:
    '''
    converts nd array pose from factor graph data to
    sym.pose3 type
    '''
    # assuming pose in this sequence : tx, ty, tz, qx, qy, qz, qw .
    #
    q = pose.rotation().toQuaternion()  # gtsam type
    rotation = sym.Rot3(np.array([q.x(), q.y(), q.z(), q.w()]))
    translation = np.array([pose.x(), pose.y(), pose.z()])
    return sym.Pose3(R=rotation, t=translation)


def convert_to_sym_camera(cam_intrinsics: np.ndarray) -> sym.LinearCameraCal:
    '''
    converts the intrinsic parameters into sym.LinearCameraCal type
    '''
    f_length = cam_intrinsics[:2]
    principal_pt = cam_intrinsics[2:]
    K = sym.LinearCameraCal(
        focal_length=f_length,
        principal_point=principal_pt)
    return K


def droid_slam_error_func(
        dst_img_coords: np.ndarray,
        src_img_coords: np.ndarray,
        cam_intrinsics: np.ndarray,
        this: gtsam.CustomFactor,
        v: gtsam.Values,
        H: Optional[List[np.ndarray]]) -> np.ndarray:
    '''
    droid slam custom factor definition.
    '''
    pose_i_key = this.keys()[0]
    pose_j_key = this.keys()[1]
    depth_key = this.keys()[2]
    print(f'Keys accessed : pose_i - {pose_i_key}, pose_j - {pose_j_key},\
            depth_key - {depth_key}')
    w_pose_i, w_pose_j = v.atPose3(pose_i_key), v.atPose3(pose_j_key)
    depth = v.atDouble(depth_key)
    sym_pose_i = gtsam2sym_pose(w_pose_i)
    sym_pose_j = gtsam2sym_pose(w_pose_j)
    sym_camera = convert_to_sym_camera(cam_intrinsics)
    (error, jacobian, hessian, rhs) = \
        droid_autogen.droid_slam_residual_single_factor(
        dst_img_coords,
        src_img_coords,
        d_src=depth,
        w_pose_i=sym_pose_i,
        w_pose_j=sym_pose_j,
        K=sym_camera,
        epsilon=0.01)
    # print(f'error={error}, {type(error), error.shape}')
    # print(f'jacobian={jacobian}, {type(jacobian), jacobian.shape}')
    # print(f'hessian={hessian}, {type(hessian), hessian.shape}')
    # print(f'rhs={rhs}, {type(rhs), rhs.shape}')
    if H is not None:
        H[0] = jacobian[:, :6]
        H[1] = jacobian[:, 6:12]
        H[2] = jacobian[:, 12]

    return error


def factor_graph_image_pair(
        i: int,
        j: int,
        pair_id: int,
        pose_i: Union[np.ndarray, torch.Tensor],
        pose_j: Union[np.ndarray, torch.Tensor],
        depth: Union[np.ndarray, torch.Tensor],
        weights: Union[np.ndarray, torch.Tensor],
        target_pt: Union[np.ndarray, torch.Tensor],
        intrinsics: Union[np.ndarray, torch.Tensor]) -> gtsam.NonlinearFactorGraph:
    '''
    generates the factor graph of one image pair

    :param weights: weights(i->j) assigned for each pixel in the given each image pair
    '''
    # todo check sizes of each matrix
    ROWS = depth.size()[0]
    COLS = depth.size()[1]
    symbol_xi = gtsam.symbol('x', i)
    symbol_xj = gtsam.symbol('x', j)
    if not init_values.exists(symbol_xi):
        pose_i_gtsam = convert_to_gtsam_pose(pose_i.numpy())
        init_values.insert(symbol_xi, pose_i_gtsam)
    if not init_values.exists(symbol_xj):
        pose_j_gtsam = convert_to_gtsam_pose(pose_j.numpy())
        init_values.insert(symbol_xj, pose_j_gtsam)

    graph = gtsam.NonlinearFactorGraph()
    # Poses are to be optimized.
    # X_i and X_j so they are assigned symbols.
    count_symbol = 0
    for (row, d_row) in enumerate(depth):
        for (col, d) in enumerate(d_row):
            # each depth in ith camera has to be assigned a symbol
            # as it will be optimized as a variable.

            symbol_di = gtsam.symbol('d', ROWS * COLS * i + count_symbol)

            if not init_values.exists(symbol_di):
                init_values.insert(symbol_di, depth[row, col].numpy())
            # define noise for each pixel from confidence map or weight
            # matrix
            di = depth[row, col]
            if depth[row, col].numpy() < 0.25:
                w = np.array([10e10, 10e10])
            else:
                w = 0.001 * weights[:, row, col].numpy().reshape(2, 1)
            pixel_noise_model = gtsam.noiseModel.Diagonal.Sigmas(1 / w)
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
                )
            )
            graph.add(custom_factor)
            count_symbol += 1
    return graph


def build_factor_graph(fg_data: dict) -> gtsam.NonlinearFactorGraph:
    '''
    build factor graph from complete data
    '''
    graph_data = fg_data['graph_data']
    depth = fg_data['disps']
    poses = fg_data['poses']
    weights = fg_data['c_map']
    predicted = fg_data['predicted']
    K = fg_data['intrinsics']
    ii = graph_data['ii']
    jj = graph_data['jj']
    pair_unique_id = {}
    n = 5  # ii.size()[0]
    unique_id = 0
    full_graph = gtsam.NonlinearFactorGraph()
    print(f'''Graph index - {graph_data['ii'].size()},
              poses - {poses.size()},
              ---------------------------
              weights - shape = {weights.size()},
              ---------------------------
              predicted - shape = {predicted.size()},
              ---------------------------
              depth - shape = {depth.size()},
        ''')
    for (index, (ix, jx)) in enumerate(zip(ii[:n], jj[:n])):
        key = (ix, jx)
        if key not in pair_unique_id.keys():
            pair_unique_id[key] = unique_id
            unique_id += 1
        if max(ix, jx).item() > poses.size()[0] - 1:
            print(f"Ignoring index - {ix , jx} - out of bounds")
            continue
        print(f'Index - {index} - Adding factors for {ix} - {jx} edge')
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
    print(f'Number of factors in full factor graph = {full_graph.nrFactors()}')
    return full_graph


if __name__ == "__main__":

    fg_dir = Path(
        "/media/jagatpreet/D/datasets/uw_rig/samples"
    ).joinpath(
        "woodshole_east_dock_1/factorgraph_data_2023_11_27_16_10_29"
    )
    if fg_dir.exists():
        files_list = sorted(os.listdir(fg_dir))
    print(f'Number of files = {len(files_list)}')
    fg_file = fg_dir.joinpath(files_list[0])

    # Prior noise definition for first two poses.
    #  3D rotational standard deviation of prior factor - gaussian model
    #  (degrees)
    prior_rpy_sigma = 1
    # 3D translational standard deviation of of prior factor - gaussian model
    # (meters)
    prior_xyz_sigma = 0.05
    sigma_angle = np.deg2rad(prior_rpy_sigma)
    prior_noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([sigma_angle,
                                                                   sigma_angle,
                                                                   sigma_angle,
                                                                   prior_xyz_sigma,
                                                                   prior_xyz_sigma,
                                                                   prior_xyz_sigma]
                                                                  )
                                                         )
    symbol_first_pose = gtsam.symbol('x', 0)
    symbol_second_pose = gtsam.symbol('x', 1)
    print(f'Analyzing file : { fg_file}')
    fg_data = import_fg_from_pickle_file(fg_file)
    print_factor_graph_stats(fg_data)
    graph = build_factor_graph(fg_data)
    graph.push_back(
        gtsam.PriorFactorPose3(symbol_first_pose,
                               init_values.atPose3(symbol_first_pose),
                               prior_noise_model)
    )
    graph.push_back(
        gtsam.PriorFactorPose3(symbol_second_pose,
                               init_values.atPose3(symbol_first_pose),
                               prior_noise_model)
    )

    print(f'Number of factors={graph.nrFactors()}')
    number_of_iters = input(
        'Enter integer number of iterations for optimization:')
    print(f'Number of iterations {number_of_iters}')
    time.sleep(2)
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(int(number_of_iters))
    print(f" LM params: {params}")
    optimizer = gtsam.LevenbergMarquardtOptimizer(
        graph, init_values, params)
    result = optimizer.optimize()
    print(f'Final result :\n {result}')
    flag = input("Linearize graph initial values: 0-> No, 1-> yes")
    jac_list = []
    b_list = []
    cov_list = []
    info_list = []
    if int(flag):
        print(f'Errors init values = {graph.error(init_values)}')
        lin_graph1 = graph.linearize(init_values)
        jac, b = lin_graph1.jacobian()
        cov = np.linalg.inv(jac.transpose() @ jac)
        info = jac.transpose() @ jac
    jac_list.append(jac)
    b_list.append(b)
    info_list.append(info)
    cov_list.append(cov)
    flag = input("Linearize graph final values: 0-> No, 1-> yes")
    if int(flag):
        print(f'Errors final values = {graph.error(result)}')
        lin_graph2 = graph.linearize(result)
        jac, b = lin_graph2.jacobian()
        cov = np.linalg.inv(jac.transpose() @ jac)
        info = jac.transpose() @ jac
    jac_list.append(jac)
    b_list.append(b)
    info_list.append(info)
    cov_list.append(cov)
    if (int(flag)):
        marginals_init = gtsam.Marginals(graph, init_values)
        marginals_new = gtsam.Marginals(graph, result)
