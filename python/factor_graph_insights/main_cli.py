"""
Main command line interface applications 
"""
from typing import Union, Dict, List
import logging
import argparse
import signal
import time
from pathlib import Path
import sys
import os
import numpy as np
from factor_graph_insights.visualization_utils import CovTrendsVisualizer 
from factor_graph_insights.process_data import FactorGraphFileProcessor

def signal_handler(sig, frame):
    print("Pressed Ctrl + C. Exiting")
    sys.exit(0)

def common_command_line_arguments(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    """
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
    ap.add_argument("--raw_image_folder", type=str, default = Path("/data/euroc/"),
                    help="save location of images in a factor graph")
    return ap

def handle_file():
    """"""
    signal.signal(signal.SIGINT, signal_handler)
    ap = argparse.ArgumentParser("Argument parser for factor graph file analysis")
    ap = common_command_line_arguments(ap)
    ap.add_argument("-f","--filepath", required=True, type=Path, help="filepath of the factor graph file")
    
    # ap.add_argument("--stride", type=int, default=2, help="images to skip when computing ids")
    
    args = ap.parse_args()
    fg_file_processor = FactorGraphFileProcessor(args)
    fg_file_processor.initialize_ba_problem()
    success = fg_file_processor.process()
    adj_matrix = fg_file_processor.covisibility_graph_to_adj_matrix()
    
    # viz
    visualizer = CovTrendsVisualizer(fg_file_processor.metadata)
    
    visualizer.plot_trajectory(
        fg_file_processor.ba_problem.poses, metadata=fg_file_processor.metadata)
    visualizer.plot_trends_determinants(fg_file_processor.D_cat, adj_matrix)
    visualizer.plot_trends_singular_values(fg_file_processor.S_cat)
    

def animate():
    """"""
    
def handle_batch():
    """"""
    signal.signal(signal.SIGINT, signal_handler)
    ap = argparse.ArgumentParser("Argument parser for factor graph analysis")
    ap = common_command_line_arguments(ap)
    ap.add_argument("-d","--dir", 
                    type=Path, help="folderpath with factorgraphs")
    ap.add_argument("-s","--start_id", 
                    default = 0, type=int, help="index of first factor graph file")
    ap.add_argument("-e","--end_id", 
                    default = -1, type=int, help="index of last factor graph file")
    ap.add_argument("--pause", 
                    action="store_true", help="pause the plot to show")
    ap.add_argument("--save_dir", 
                    type=str, default = Path("/data/jagat/processed/plots"),
                    help="save location for the factor graph")
    
    args = ap.parse_args()
    
    start = args.start_id
    end = args.end_id
    fg_dir = Path(args.dir)
    if fg_dir.exists():
        files_list = sorted(os.listdir(fg_dir))
    else:
        raise ValueError("factor graph folder does not exist")
    
    logging.info(
            f"Number of files to process in {fg_dir} = {len(files_list[start:end])}"
    )
    files_that_worked = []
    files_that_failed = []
    fg_file_processor = FactorGraphFileProcessor(args)
    
    # process the batch
    for relative_index, filename in enumerate(files_list[start : end]):
        logging.info(f"Files worked : {start + relative_index}")
        fg_file = fg_dir.joinpath(filename)
        # set current factor graph file
        fg_file_processor.factor_graph_file = fg_file
        # process individual file
        fg_file_processor.initialize_ba_problem()
        success = fg_file_processor.process()
        if success:
            files_that_worked.append(int(filename.split("_")[1]))
        else:
            files_that_failed.append(int(filename.split("_")[1]))
    
    # visualizer = CovTrendsVisualizer(fg_folder_processor.)
    # if args.plot:
    #     Visualizer.plot_pose_covariance(S_cat, D_cat, self.metadata)

