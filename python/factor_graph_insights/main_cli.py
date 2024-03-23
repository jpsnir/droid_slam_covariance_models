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
import numpy as np
from factor_graph_insights.visualization_utils import BaseVisualizer 
from factor_graph_insights.process_data import FactorGraphFileProcessor, FgFolderProcessor

def signal_handler(sig, frame):
    print("Pressed Ctrl + C. Exiting")
    sys.exit(0)

def visualize_factor_graphs():
    """"""
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
    fg_file_processor = FactorGraphFileProcessor(args)
    fg_file_processor.initialize_ba_problem()
    success = fg_file_processor.process()
    adj_matrix = fg_file_processor.covisibility_graph_to_adj_matrix()
    BaseVisualizer.plot_trajectory(
        fg_file_processor.ba_problem.poses, metadata=fg_file_processor.metadata)
    

def animate():
    """"""
    
def process_factor_graphs():
    """"""
    signal.signal(signal.SIGINT, signal_handler)
    ap = argparse.ArgumentParser("Argument parser for factor graph analysis")
    ap.add_argument("-d","--dir", 
                    type=Path, help="folderpath with factorgraphs")
    ap.add_argument("-s","--start_id", 
                    default = 0, type=int, help="index of first factor graph file")
    ap.add_argument("-e","--end_id", 
                    default = -1, type=int, help="index of last factor graph file")
    ap.add_argument("-p","--plot", 
                    action="store_true", help="plot graphs")
    ap.add_argument("--number_of_edges", 
                    type=int, default = -1,help="number of edges to process")
    ap.add_argument("--near_depth_threshold", 
                    type=float, default = 0.25, 
                    help="reject points that are closer than this distance")
    ap.add_argument("--far_depth_threshold", 
                    type=float, default = 4, 
                    help="reject points that are farther than this distance")
    ap.add_argument("--loglevel", 
                    default="info", help="provide loglevel")
    ap.add_argument("--pause", 
                    action="store_true", help="pause the plot to show")
    ap.add_argument("--save_dir", 
                    type=str, default = Path("/data/jagat/processed/plots"),
                    help="save location for the factor graph")
    
    args = ap.parse_args()
    
    if args.plot:
        Visualizer.plot_pose_covariance(S_cat, D_cat, self.metadata)

