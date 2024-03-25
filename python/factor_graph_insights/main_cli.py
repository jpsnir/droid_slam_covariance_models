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
from factor_graph_insights.visualization_utils import BaseVisualizer, CovTrendsVisualizer
from factor_graph_insights.process_data import FactorGraphFileProcessor
from factor_graph_insights.animations import CovarianceResultsAnimator

def signal_handler(sig, frame):
    print("Pressed Ctrl + C. Exiting")
    sys.exit(0)


def common_command_line_arguments(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    """
    ap.add_argument("-p", "--plot", action="store_true", help="plot graphs")
    ap.add_argument("--number_of_edges", type=int, default=-
                    1, help="number of edges to process")
    ap.add_argument("--near_depth_threshold", type=float, default=0.25,
                    help="reject points that are closer than this distance")
    ap.add_argument("--far_depth_threshold", type=float, default=3,
                    help="reject points that are farther than this distance")
    ap.add_argument("--loglevel", default="info", help="provide loglevel")
    ap.add_argument("--log_to_file", action="store_true",
                    help="save logs in a log file")
    ap.add_argument("--save_dir", type=str, default=Path("/data/jagat/processed/fg_images"),
                    help="save location of images in a factor graph")
    
    return ap


def handle_file():
    """"""
    signal.signal(signal.SIGINT, signal_handler)
    ap = argparse.ArgumentParser(
        "Argument parser for factor graph file analysis")
    ap = common_command_line_arguments(ap)
    ap.add_argument("-f", "--filepath", required=True, type=Path,
                    help="filepath of the factor graph file")

    # ap.add_argument("--stride", type=int, default=2, help="images to skip when computing ids")

    args = ap.parse_args()
    args.raw_image_folder = None
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

def images_in_factor_graph():
    signal.signal(signal.SIGINT, signal_handler)
    ap = argparse.ArgumentParser(
        "Argument parser for factor graph file analysis")
    ap.add_argument("--raw_image_folder", type=Path, default=Path("/data/euroc/"),
                    help="save location of images in a factor graph")
    ap.add_argument("-f", "--filepath", required=True, type=Path,
                    help="filepath of the factor graph file")
    args = ap.parse_args()
    args.plot = True
    args.number_of_edges = -1
    args.near_depth_threshold = 0.25
    args.far_depth_threshold = 2
    args.save_dir = None
    args.loglevel = "info"
    args.log_to_file = False
    
    fg_file_processor = FactorGraphFileProcessor(args)
    fg_file_processor.image_and_node_ids()
    visualizer = BaseVisualizer(fg_file_processor.metadata)
    image_file_paths, node_ids, kf_ids = fg_file_processor.extract_kf_images()
    visualizer.visualize_images(image_file_paths)
    

def animate():
    """"""
    signal.signal(signal.SIGINT, signal_handler)
    ap = argparse.ArgumentParser(
        "Argument parser for factor graph file analysis")
    ap = common_command_line_arguments(ap)
    ap.add_argument("-f", "--filepath", required=True, type=Path,
                    help="filepath of the factor graph file")
    ap.add_argument("--raw_image_folder", type=Path, default=Path("/data/euroc/"),
                    help="save location of images in a factor graph")
    args = ap.parse_args()
    args.plot = False
    
    fg_file_processor = FactorGraphFileProcessor(args)
    fg_file_processor.initialize_ba_problem()
    success = fg_file_processor.process()
    adj_matrix = fg_file_processor.covisibility_graph_to_adj_matrix()
    image_file_paths, node_ids, kf_ids = fg_file_processor.extract_kf_images()
    animation_data = {
        "image_paths" : image_file_paths,
        "adj_matrix": adj_matrix,
        "cov_determinants": fg_file_processor.D_cat,
        "trajectory" : fg_file_processor.ba_problem.poses[[node_ids]],
        "kf_ids" : kf_ids,
    }
    animator = CovarianceResultsAnimator()
    animator.set_data(animation_data)
    animator.run(show=False)
    


def list_of_ints(arg: str) -> List[int]:
    """
    custom argument parser
    returns a list of integers from a comma separated integer list
    """
    l_ints = []
    try:
        l_ints = list(map(int, arg.split(",")))
    except:
        raise ValueError("Input is invalid. Cannot parse")

    return l_ints


def handle_batch():
    """"""
    signal.signal(signal.SIGINT, signal_handler)
    ap = argparse.ArgumentParser("Argument parser for factor graph analysis")
    ap = common_command_line_arguments(ap)
    ap.add_argument("-d", "--dir", required=True,
                    type=Path, help="folderpath with factorgraphs")
    ap.add_argument("-s", "--start_id",
                    default=0, type=int, help="index of first factor graph file")
    ap.add_argument("-e", "--end_id",
                    default=-1, type=int, help="index of last factor graph file")
    ap.add_argument("-list_of_files", type=list_of_ints, default=[],
                    help="comma separted list of factor files to process in the folder ")
    ap.add_argument("--pause",
                    action="store_true", help="pause the plot to show")
   
    args = ap.parse_args()

    fg_dir = Path(args.dir)
    if fg_dir.exists():
        files_list = sorted(os.listdir(fg_dir))
    else:
        raise ValueError("factor graph folder does not exist")

    # select which files to process

    if args.list_of_files:
        l_ints = sorted([args.list_of_files])
        start = l_ints[0]
        end = l_ints[-1]
        subset = files_list[[l_ints]]
    else:
        start = args.start_id
        end = args.end_id
        subset = files_list[start:end]

    logging.info(f"Number of files to process in {fg_dir} = {len(subset)}")
    files_that_worked = []
    files_that_failed = []
    fg_file_processor = FactorGraphFileProcessor(args)

    # process the batch
    for relative_index, filename in enumerate(subset):
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