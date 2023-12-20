"""
Project: factor_graph_insights
Description:
File Created: Monday, 18th December 2023 7:42:34 pm
Author: Jagatpreet (nir.j@northeastern.edu)
-----
Last Modified: Monday, 18th December 2023 7:42:34 pm
Modified By: Jagatpreet (nir.j@northeastern.edu>)
-----
Copyright <<projectCreationYear>> - 2023 Northeastern University Field Robotics Lab, Northeastern University Field Robotics Lab
"""
from factor_graph_insights.fg_builder import factor_graph_image_pair
import numpy as np
import os
import gtsam
from pathlib import Path
import pickle
from typing import Union, Dict, List
import logging


class FactorGraphData:
    @staticmethod
    def load_from_pickle_file(filename: Union[str, Path]) -> Dict:
        """
        generates parsed data from pickle file
        """
        fg_data = {}
        with open(filename, "rb") as f:
            fg_data = pickle.load(f)

        return fg_data

    @staticmethod
    def log_factor_graph_stats(fg_data: dict):
        """
        prints some basic details of the factor graph for
        error checking
        """
        print(f"data keys: {fg_data.keys()}")
        print(f"id: {fg_data['id']}")
        print(f"intrinsics: {fg_data['intrinsics']}")
        print("------------------------------")
        print(f"graph data - ii: {fg_data['graph_data']['ii']}")
        print(f"graph data - jj: {fg_data['graph_data']['jj']}")
        print(
            f"graph data - number of connections:\
            {fg_data['graph_data']['ii'].size().numel()}"
        )
        print(f"weights size: {fg_data['c_map'].size()}")
        print(f"target pts size: {fg_data['predicted'].size()}")
        print("------------------------------")
        print(f"tstamps size: {fg_data['tstamp'].size()}")
        print(f"poses size: {fg_data['poses'].size()}")
        print(f"disparity size: {fg_data['disps'].size()}")


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
    fg_data = FactorGraphData.load_from_pickle_file(fg_file)
    FactorGraphData.log_factor_graph_stats(fg_data)
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
