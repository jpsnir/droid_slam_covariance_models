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

import numpy as np
import os
import gtsam
from pathlib import Path
from typing import Union, Dict, List
import logging
from factor_graph_insights.ba_problem import BAProblem, FactorGraphData
from factor_graph_insights.graph_analysis import GraphAnalysis

if __name__ == "__main__":
    N = 5
    fg_dir = Path("/media/jagatpreet/D/datasets/uw_rig/samples").joinpath(
        "woodshole_east_dock_1/factorgraph_data_2023_12_27_14_14_35"
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
    fg_data = FactorGraphData.load_from_pickle_file(fg_file)
    ba_problem = BAProblem(fg_data)
    graph = ba_problem.build_visual_factor_graph(prior_noise_model)
    init_vals = ba_problem.i_vals
    analyzer = GraphAnalysis(graph)

    info, marginals = analyzer.marginals(init_vals)
    print(f" Rank of matrix {np.linalg.matrix_rank(info)}")
