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

def signal_handler(sig, frame):
    print("Pressed Ctrl + C. Exiting")
    sys.exit(0)

def visualize_factor_graphs():
    """"""
    signal.signal(signal.SIGINT, signal_handler)
    ap = argparse.ArgumentParser("Argument parser for factor graph analysis")
    ap.add_argument("-f","--filepath", 
                    required=True, type=Path, 
                    help="filepath of the factor graph file")
    ap.add_argument("-p","--plot", 
                    action="store_true", help="plot graphs")
    ap.add_argument("--number_of_edges", 
                    type=int, default = -1,help="number of edges to process")
    ap.add_argument("--near_depth_threshold", 
                    type=float, default = 0.25, 
                    help="reject points that are closer than this distance")
    ap.add_argument("--far_depth_threshold", 
                    type=float, default = 3, 
                    help="reject points that are farther than this distance")
    ap.add_argument("--loglevel", 
                    default="info", help="provide loglevel")
    ap.add_argument("--log_to_file", 
                    action="store_true", help="save logs in a log file")
    ap.add_argument("--save_dir", 
                    type=str, default = Path("/data/jagat/processed/fg_images"),
                    help="save location of images in a factor graph")

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
    
    
    if self.plot:
                plot_pose_covariance(S_cat, D_cat, self.metadata)

