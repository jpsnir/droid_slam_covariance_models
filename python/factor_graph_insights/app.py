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
import argparse
from factor_graph_insights.ba_problem import BAProblem, FactorGraphData
from factor_graph_insights.graph_analysis import GraphAnalysis
from matplotlib import pyplot as plt
import signal
import sys
import time 

log_levels = {
    'critical': logging.CRITICAL,
    'error': logging.ERROR,
    'warn': logging.WARNING,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG
}

def signal_handler(sig, frame):
    print("Pressed Ctrl + C. Exiting")
    sys.exit(0)

def plot_pose_covariance(
    singular_values: np.array,
    Determinant: np.array = None,
    image_ids: np.array = None,
    file_num:int = None,
    save_location:Union[str, Path] = None,
    pause:bool = False,
):
    """plots the covariance of poses for n dimensional pose
       from the singular values of the marginal covariance matrix.

    Args:
        singular_values (np.array): _description_
    """
    
    # if fig_handles is None:
    #     fig1, fig2, fig3 = None, None, None
    # else:
    #     fig1, fig2, fig3 = fig_handles

    if isinstance(save_location, str):
        save_location = Path(save_location)
    
    M, N = singular_values.shape
    if image_ids is not None:
        x_ = image_ids    
    else:
        x_ = range(0, M)
    labels = ["roll", "pitch", "yaw"]

        
    fig1, ax_theta = plt.subplots(3, 1, figsize=(12, 12))
    for i, ax in enumerate(ax_theta):
        ax.plot(x_, singular_values[:, i], "--*")
        ax.set_xlabel("Pose id ", fontsize=18)
        ax.set_ylabel(f"{labels[i]}", fontsize=18)
        ax.grid(visible=True)
    fig1.suptitle(
        f"Trends of singular Values of covariance matrix - angles - {file_num}", fontsize=18
    )
    
    # Singular values
    fig2, ax_position = plt.subplots(3, 1, figsize=(12, 12))
    labels = ["x(m)", "y(m)", "z(m)"]
    for i, ax in enumerate(ax_position):
        ax.plot(x_, singular_values[:, i + 3], "--*")
        ax.set_xlabel("Pose id ", fontsize=18)
        ax.set_ylabel(f"{labels[i]}", fontsize=18)
        ax.grid(visible=True)
    fig2.suptitle(
        f"Trends of singular Values of covariance matrix - positions - {file_num}]", fontsize=18
    )
    
    # Determinanant
    if Determinant is not None:
        fig3, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.plot(x_, Determinant, "--*")
        ax.set_yscale("log")
        ax.set_xlabel("Pose id ", fontsize=18)
        ax.set_ylabel(f"Determinant of Sigma (Log space)", fontsize=18)
        ax.grid(visible=True)
        fig3.suptitle(f"Trend of determinants of covariance matrix - {file_num}", fontsize=18)
    
    if save_location is not None:      
        fig1.savefig(save_location.joinpath(f"angle_{file_num}.png"))
        fig2.savefig(save_location.joinpath(f"position_{file_num}.png"))
        fig3.savefig(save_location.joinpath(f"det_{file_num}.png"))
        logging.info("Saving plots")
    
    if pause:
            plt.show(block=True)
    else: 
            plt.close(fig1)
            plt.close(fig2)
            plt.close(fig3)
    return [fig1, fig2, fig3]

def visualize_images(image_id_list: List):
    """"""
    

def plot_error_from_groundtruth(image_id_list: List, covariance: List ):
    """"""
    

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    ap = argparse.ArgumentParser("Argument parser for factor graph analysis")
    ap.add_argument("-d","--dir", type=Path, help="folderpath with factorgraphs")
    ap.add_argument("-s","--start_id", default = 0, type=int, help="index of first factor graph file")
    ap.add_argument("-e","--end_id", default = -1, type=int, help="index of last factor graph file")
    ap.add_argument("-p","--plot", action="store_true", help="plot graphs")
    ap.add_argument("--number_of_edges", type=int, default = -1,help="number of edges to process")
    ap.add_argument("--near_depth_threshold", type=float, default = 0.25, 
                    help="reject points that are closer than this distance")
    ap.add_argument("--far_depth_threshold", type=float, default = 4, 
                    help="reject points that are farther than this distance")
    ap.add_argument("--loglevel", default="info", help="provide loglevel")
    ap.add_argument("--pause", action="store_true", help="pause the plot to show")
    ap.add_argument("--save_dir", type=str, default = Path("/data/jagat/processed/plots"),
                    help="save location for the factor graph")
    
    args = ap.parse_args()
    fg_dir = Path(args.dir)
    parent_path = fg_dir.parents[3]
    rel_path = fg_dir.relative_to(parent_path)
    # plot save location 
    plot_folder_name = str(rel_path).replace("/","_")
    plot_save_location = Path(args.save_dir).joinpath(plot_folder_name)
    if not plot_save_location.exists(): 
        plot_save_location.mkdir(parents=True)
    logging.basicConfig(filename=str(plot_save_location.joinpath("app.log")), level=args.loglevel.upper())
    logging.info(f"logging level = {logging.root.level}")
    logging.info(f"plots will be saved at : {plot_save_location.absolute()}")
    
    start = args.start_id
    end = args.end_id
    if fg_dir.exists():
        files_list = sorted(os.listdir(fg_dir))
    logging.info(f"Number of files to processed in {fg_dir} = {len(files_list[start:end])}")
    files_that_worked = []
    files_that_failed = []
    
    
    # analyzing all factor graph files within the start and  end range.
    for file_num, filename in enumerate(files_list[start:end]):
        fg_file = fg_dir.joinpath(filename)
        fg_data = FactorGraphData.load_from_pickle_file(fg_file)
        ba_problem = BAProblem(fg_data)
        ba_problem.near_depth_threshold = args.near_depth_threshold
        ba_problem.far_depth_threshold = args.far_depth_threshold

        oldest_nodes = ba_problem.get_oldest_poses_in_graph()
        prior_definition = ba_problem.set_prior_definition(pose_indices=oldest_nodes)
        logging.debug(f"Prior definition : {prior_definition}")
        ba_problem.add_visual_priors(prior_definition)
        if args.number_of_edges > 0:
            graph = ba_problem.build_visual_factor_graph(N_edges=args.number_of_edges)
        else:
            graph = ba_problem.build_visual_factor_graph(N_edges=ba_problem.edges)
        init_vals = ba_problem.i_vals
        S_cat = np.zeros([len(ba_problem.poses), 6])
        D_cat = np.zeros([len(ba_problem.poses), 1])
        node_ids_i = ba_problem.get_node_ids(args.number_of_edges)
        image_ids = ba_problem.get_image_ids()
        logging.info(f"Prior node ids: {oldest_nodes}")
        logging.info(f"node ids - i: size: {len(node_ids_i)} - {node_ids_i}")
        logging.info(f"node timestamps: size : {len(image_ids)} - {image_ids}")

        try:
            analyzer = GraphAnalysis(graph)
            marginals = analyzer.marginals(init_vals)
            logging.info(f"Files worked : {start + file_num}")

            symbols = lambda indices: [gtsam.symbol("x", idx) for idx in indices]
            cov_list = []
            for p_id in node_ids_i:
                cov = marginals.marginalCovariance(gtsam.symbol("x", p_id))
                cov_list.append(cov)
                U, S, V = np.linalg.svd(cov)
                S_cat[p_id, :] = S
                d = np.linalg.det(cov)
                D_cat[p_id, :] = d
                logging.debug(f" marginal covariance = {cov}")
                logging.debug(f" singular values at {p_id} = {S}")
                logging.debug(f" determinant at {p_id} = {d}")
                logging.debug(f"Concatenated singular values = {S_cat}")
                logging.debug(f"Concatenated covariance list : {cov_list}")
            if args.plot:
                plot_pose_covariance(S_cat, D_cat, image_ids, file_num, plot_save_location, args.pause )
                
            files_that_worked.append(int(filename.split("_")[1]))
        except Exception as e:
            logging.error(f"File number {file_num}, Filename - {filename}, error code - {e}")
            files_that_failed.append(int(filename.split("_")[1]))
            time.sleep(2)
    logging.info(f"files that worked : {files_that_worked}")
    logging.info(f"files that failed : {files_that_failed}")
    
    
    
    # print(f" Rank of matrix {np.linalg.matrix_rank(info)}")
