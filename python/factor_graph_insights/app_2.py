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
import matplotlib.gridspec as gridspec
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    adjacency_matrix_data: Dict = None,
    metadata: Dict = None,
    image_ids: np.array = None,
    file_num:int = None,
    save_location:Union[str, Path] = None,
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

    if metadata is not None:
        save_location = Path(metadata['plot_save_location'])
        node_ids = metadata['node_ids']
        image_ids = metadata['image_ids']
        x_ = image_ids[[node_ids]]
        file_num = metadata['file_id']
        
    M, N = singular_values.shape
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
        fig3 = plt.figure(figsize=(20, 6))
        gs = fig3.add_gridspec(1, 2, width_ratios=[2, 1])
        ax = [] 
        ax.append(fig3.add_subplot(gs[0]))
        ax.append(fig3.add_subplot(gs[1]))
        ax[0].plot(x_, Determinant, "--*")
        for i, (xi, yi) in enumerate(zip(x_, Determinant)):
            ax[0].annotate(str(i), (xi,yi), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=10, rotation=45)
        ax[0].set_yscale("log")
        ax[0].set_xlabel("Pose id ", fontsize=18)
        ax[0].set_xticks(image_ids[[node_ids]])
        ax[0].xaxis.set_ticks(image_ids[[node_ids]])
        ax[0].tick_params(axis="both", which='major', 
                          labelsize=8, labelrotation=60)
        # ax2 = ax[0].twiny()
        # ax2.set_xlabel("ID in adjacency graph")
        #ax[0].yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=100))
        ax[0].set_ylabel(f"Determinant of \nrelative marginal covariances of poses \n(Log space)", fontsize=14)
        ax[0].minorticks_on()
        ax[0].grid(visible=True, which='both', linestyle=":")
        ax[0].set_title(f"{metadata['dataset_name']} - {len(node_ids)} keyframes - {int(image_ids[-1].item()/20)}s", fontsize=18)
        
        M = adjacency_matrix_data['M']
        node_ids = node_ids
        ax[1].spy(M[:len(node_ids), :len(node_ids)])
        ax[1].xaxis.set_ticks(node_ids - node_ids[0])
        ax[1].yaxis.set_ticks(node_ids - node_ids[0])
        ax[1].tick_params(axis='both', which='major', labelsize=10)
        ax[1].set_title("Adjaceny graph of the bundle adjustment problem")
        plt.tight_layout()
    if metadata['plot']:
        plt.show()
    
    if save_location is not None:      
        fig1.savefig(save_location.joinpath(f"angle_{file_num}.png"))
        fig2.savefig(save_location.joinpath(f"position_{file_num}.png"))
        fig3.savefig(save_location.joinpath(f"{metadata['dataset_name']}.pdf"))
        logging.info("Saving plots")
        
def visualize_images(image_id_list: List):
    """"""
    
    
    

def plot_error_from_groundtruth(image_id_list: List, covariance: List ):
    """"""


def plot_coordinate_frame(ax, R, origin=[0, 0, 0], size=1):
    """
    Plot a coordinate frame defined by the rotation matrix R.
    
    Parameters:
        ax (Axes3D): Matplotlib 3D axis object.
        R (numpy.ndarray): 3x3 rotation matrix.
        origin (list): Origin point of the coordinate frame.
        size (float): Size of the coordinate frame axes.
    """
    axes = size * R
    ax.quiver(origin[0], origin[1], origin[2], axes[0, 0], axes[1, 0], axes[2, 0], color='r', label='X')
    ax.quiver(origin[0], origin[1], origin[2], axes[0, 1], axes[1, 1], axes[2, 1], color='g', label='Y')
    ax.quiver(origin[0], origin[1], origin[2], axes[0, 2], axes[1, 2], axes[2, 2], color='b', label='Z')

def plot_trajectory(poses: np.array, metadata:dict):
    """"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print(poses)
    node_ids = metadata['node_ids']
    poses = poses[[node_ids]]
    ax.plot(poses[:, 0], poses[:, 1], poses[:, 2],'-*')
    ax.plot(poses[0, 0], poses[0, 1], poses[0, 2], 'rX', )
    from factor_graph_insights.fg_builder import DataConverter
    
    for i, p in enumerate(poses):
        ax.text(poses[i, 0], poses[i, 1], poses[i, 2], f"{i}", color = "black",va="bottom")
        pose_w_c = DataConverter.to_gtsam_pose(poses[i]).inverse()
        T = pose_w_c.matrix()
        plot_coordinate_frame(ax, T[:3, :3], 
                              origin=[poses[i, 0], poses[i, 1], poses[i, 2]], size=0.05)
        
    ax.text(poses[0, 0], poses[0, 1], poses[0, 2], "S", color = "red", va="bottom")
    ax.text(poses[-1, 0], poses[-1, 1], poses[-1, 2], "E", color = "green",va="bottom")
    if metadata['plot_save_location'] is not None:
        fig.savefig(metadata['plot_save_location'].joinpath(f"factor_graph_trajectory.pdf"))    

def covisibility_graph_to_adj_matrix(src_nodes, dst_nodes):
    """"""
    
    node_ids = list(set(np.sort(src_nodes.numpy())))
    N = len(node_ids)
    M = np.zeros((N, N))
    for src, dst in zip(src_nodes, dst_nodes):
        for i, n in enumerate(node_ids):
            if(src.item()==n):
                row = i
            if(dst.item()==n):
                col = i
        M[row, col] = 1
    return np.triu(M), node_ids
    
    
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    ap = argparse.ArgumentParser("Argument parser for factor graph analysis")
    ap.add_argument("-f","--filepath", required=True, type=Path, help="filepath of the factor graph file")
    ap.add_argument("-p","--plot", action="store_true", help="plot graphs")
    ap.add_argument("--number_of_edges", type=int, default = -1,help="number of edges to process")
    ap.add_argument("--near_depth_threshold", type=float, default = 0.25, 
                    help="reject points that are closer than this distance")
    ap.add_argument("--far_depth_threshold", type=float, default = 3, 
                    help="reject points that are farther than this distance")
    ap.add_argument("--loglevel", default="info", help="provide loglevel")
    ap.add_argument("--log_to_file", action="store_true", help="save logs in a log file")
    ap.add_argument("--save_dir", type=str, default = Path("/data/jagat/processed/fg_images"),
                    help="save location of images in a factor graph")
    # ap.add_argument("--stride", type=int, default=2, help="images to skip when computing ids")
    
    args = ap.parse_args()
    fg_file = Path(args.filepath)
    metadata = {}
    metadata['cam_type'] = fg_file.parents[2].stem
    metadata['dataset_name'] = fg_file.parents[3].stem
    metadata['parent_folder'] = fg_file.parents[4]
    metadata['filename'] = fg_file.stem
    metadata['file_id'] =metadata['filename'].split("_")[1]
    rel_path = fg_file.relative_to(metadata['parent_folder'])
    # plot save location 
    folder_name = str(rel_path).replace("/","_").split(".")[0]
    metadata['plot_save_location'] = Path(args.save_dir).joinpath(folder_name)
    metadata['plot'] = args.plot
    raw_image_folder=Path(f"/data/jagat/euroc/{metadata['dataset_name']}/mav0/cam0/data/")
    gt_data = Path(f"/data/jagat/processed/evo/{metadata['dataset_name']}/{metadata['cam_type']}/lba/")
    
    if not metadata['plot_save_location'].exists(): 
        metadata['plot_save_location'].mkdir(parents=True)
        
    if args.log_to_file:
        logging.basicConfig(filename=str(metadata['plot_save_location'].joinpath("app_2.log")), 
                            level=args.loglevel.upper())
    else:
        logging.basicConfig(level=args.loglevel.upper())
    
    logging.info(f"Near depth : {args.near_depth_threshold}") 
    logging.info(f"Far depth : {args.far_depth_threshold}") 
    logging.info(f"logging level = {logging.root.level}")
    logging.info(f"plots will be saved at : {metadata['plot_save_location'].absolute()}")
     
    # analyzing all factor graph files within the start and  end range.
    fg_data = FactorGraphData.load_from_pickle_file(fg_file)
    logging.info("Building bundle Adjustment problem")
    ba_problem = BAProblem(fg_data)
    
    ba_problem.near_depth_threshold = args.near_depth_threshold
    ba_problem.far_depth_threshold = args.far_depth_threshold

    oldest_nodes = ba_problem.get_oldest_poses_in_graph()
    logging.info("set priors")
    prior_definition = ba_problem.set_prior_definition(pose_indices=oldest_nodes)
    logging.debug(f"Prior definition : {prior_definition}")
    ba_problem.add_visual_priors(prior_definition)
    if args.number_of_edges > 0:
        graph = ba_problem.build_visual_factor_graph(N_edges=args.number_of_edges)
    else:
        graph = ba_problem.build_visual_factor_graph(N_edges=ba_problem.edges)
    init_vals = ba_problem.i_vals
    
    metadata['node_ids'] = ba_problem.get_node_ids(args.number_of_edges)
    metadata['image_ids'] = ba_problem.get_image_ids()
    logging.info(f"Prior node ids: {oldest_nodes}")
    logging.info(f"node ids - i: size: {len(metadata['node_ids'])} - {metadata['node_ids']}")
    logging.info(f"node timestamps: size : {len(metadata['image_ids'])} - {metadata['image_ids']}")
    
    ################
    
    adjacency_matrix, node_ids = covisibility_graph_to_adj_matrix(ba_problem.src_nodes, 
                                                        ba_problem.dst_nodes)
    adjacency_matrix_data = {'M': adjacency_matrix,
                             'node_ids': node_ids}
    plot_trajectory(ba_problem.poses, metadata=metadata)
    S_cat = np.zeros([len(metadata['node_ids']), 6])
    D_cat = np.zeros([len(metadata['node_ids']), 1])                                               
    try:
        analyzer = GraphAnalysis(graph)
        marginals = analyzer.marginals(init_vals)
        symbols = lambda indices: [gtsam.symbol("x", idx) for idx in indices]
        cov_list = []
        for i, p_id in enumerate(metadata['node_ids']):
            cov = marginals.marginalCovariance(gtsam.symbol("x", p_id))
            cov_list.append(cov)
            U, S, V = np.linalg.svd(cov)
            S_cat[i, :] = S
            d = np.linalg.det(cov)
            D_cat[i, :] = d
            logging.debug(f" marginal covariance = {cov}")
            logging.debug(f" singular values at {p_id} = {S}")
            logging.debug(f" determinant at {p_id} = {d}")
            logging.debug(f"Concatenated singular values = {S_cat}")
            logging.debug(f"Concatenated covariance list : {cov_list}")
        logging.info("Marginal covariances computation finished")
        plot_pose_covariance(S_cat, D_cat, adjacency_matrix_data, metadata)
            
    except Exception as e:
        logging.error(f"File number {metadata['file_id']}, Filename - {metadata['filename']}, error code - {e}")
        time.sleep(2)