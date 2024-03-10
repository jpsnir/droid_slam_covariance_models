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
from matplotlib import pyplot as plt


def plot_pose_covariance(
    singular_values: np.array,
    Determinant: np.array = None,
):
    """plots the covariance of poses for n dimensional pose
       from the singular values of the marginal covariance matrix.

    Args:
        singular_values (np.array): _description_
    """

    fig1, ax_theta = plt.subplots(3, 1, figsize=(12, 12))
    fig2, ax_position = plt.subplots(3, 1, figsize=(12, 12))
    M, N = singular_values.shape
    x_ = range(0, M)
    labels = ["roll", "pitch", "yaw"]
    for i, ax in enumerate(ax_theta):
        ax.plot(x_, singular_values[:, i], "--*")
        ax.set_xlabel("Pose id ", fontsize=18)
        ax.set_ylabel(f"{labels[i]}", fontsize=18)
        ax.grid(visible=True)
    fig1.suptitle(
        "Trends of singular Values of covariance matrix - angles", fontsize=18
    )
    labels = ["x(m)", "y(m)", "z(m)"]
    for i, ax in enumerate(ax_position):
        ax.plot(x_, singular_values[:, i + 3], "--*")
        ax.set_xlabel("Pose id ", fontsize=18)
        ax.set_ylabel(f"{labels[i]}", fontsize=18)
        ax.grid(visible=True)
    fig2.suptitle(
        "Trends of singular Values of covariance matrix - positions", fontsize=18
    )

    if Determinant is not None:
        fig3, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.plot(x_, Determinant, "--*")
        ax.set_yscale("log")
        ax.set_xlabel("Pose id ", fontsize=18)
        ax.set_ylabel(f"Determinant of Sigma (Log space)", fontsize=18)
        ax.grid(visible=True)
        fig3.suptitle("Trend of determinants of covariance matrix", fontsize=18)
    plt.show()


if __name__ == "__main__":
    N = 2
    fg_dir = Path("/media/jagatpreet/D/datasets/uw_rig/samples").joinpath(
        "woodshole_east_dock_1/factorgraph_data_2023_12_27_14_14_35"
    )
    if fg_dir.exists():
        files_list = sorted(os.listdir(fg_dir))
    print(f"Number of files = {len(files_list[0:N])}")

    for file_num, filename in enumerate(files_list[0:N]):
        fg_file = fg_dir.joinpath(filename)
        fg_data = FactorGraphData.load_from_pickle_file(fg_file)
        ba_problem = BAProblem(fg_data)
        oldest_nodes = ba_problem.get_oldest_poses_in_graph()
        prior_definition = ba_problem.set_prior_definition(pose_indices=oldest_nodes)
        ba_problem.add_visual_priors(prior_definition)
        graph = ba_problem.build_visual_factor_graph(N_edges=ba_problem.edges)
        init_vals = ba_problem.i_vals
        S_cat = np.zeros([len(ba_problem.poses), 6])
        D_cat = np.zeros([len(ba_problem.poses), 1])
        try:
            analyzer = GraphAnalysis(graph)
            marginals = analyzer.marginals(init_vals)
            print(f"Files worked : {file_num}")

            for p_id in range(0, len(ba_problem.poses)):
                cov = marginals.marginalCovariance(gtsam.symbol("x", p_id))
                U, S, V = np.linalg.svd(cov)
                S_cat[p_id, :] = S
                d = np.linalg.det(cov)
                D_cat[p_id, :] = d
                print(f" marginal covariance = {cov}")
                print(f" singular values at {p_id} = {S}")
                print(f" determinant at {p_id} = {d}")
        except Exception as e:
            print(f"File number {file_num}, Filename - {filename}, error code - {e}")
        print(f"Concatenated singular values = {S_cat}")
        plot_pose_covariance(S_cat, D_cat)
    # print(f" Rank of matrix {np.linalg.matrix_rank(info)}")
