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

    for i, filename in enumerate(files_list):
        fg_file = fg_dir.joinpath(filename)
        fg_data = FactorGraphData.load_from_pickle_file(fg_file)
        ba_problem = BAProblem(fg_data)
        oldest_nodes = ba_problem.get_oldest_poses_in_graph()
        prior_definition = ba_problem.set_prior_definition(pose_indices=oldest_nodes)
        ba_problem.add_visual_priors(prior_definition)
        graph = ba_problem.build_visual_factor_graph(N_edges=ba_problem.edges)
        init_vals = ba_problem.i_vals
        try:
            analyzer = GraphAnalysis(graph)
            marginals = analyzer.marginals(init_vals)
            print(f"Files worked : {i}")
            # for i in range(0, len(ba_problem.poses)):
            #     cov = marginals.marginalCovariance(gtsam.symbol("x", i))
            #     print(f" marginal covariance = {cov}")
        except Exception as e:
            print(f"File number {i}, Filename - {filename}, error code - {e}")
    # print(f" Rank of matrix {np.linalg.matrix_rank(info)}")
