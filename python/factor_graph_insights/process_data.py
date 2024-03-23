"""
process_data module contains factor graph file processing class to 
analyse the covariance of factor graph data in the file.

Also it has the factor graph folder processing class which is basically 
a wrapper over the fileprocesor. 
"""

import numpy as np
import os
import gtsam
from pathlib import Path
from typing import Union, Dict, List
import logging
from factor_graph_insights.ba_problem import BAProblem, FactorGraphData
from factor_graph_insights.graph_analysis import GraphAnalysis
import sys
import time
from typing import Tuple, Dict, List


class FactorGraphFileProcessor:
    """
    process a single factor graph file and analyze the covariance
    """

    def __init__(self, args):
        """
        initializes the factor graph processer with input arguments
        """
        
        # save locations
        self.save_dir = args.save_dir
        self.raw_image_folder = args.raw_image_folder
        if args.filepath is not None:
            self.factor_graph_file = Path(args.filepath)
        
        # ba problem parameters
        self.near_depth_threshold = args.near_depth_threshold
        self.far_depth_threshold = args.far_depth_threshold
        self.number_of_edges = args.number_of_edges
        
        # processed parameters
        self.D_cat = None
        self.S_cat = None
        self.metadata = {}
        
        # plot 
        self.metadata["plot"] = args.plot
        if not self.metadata["plot_save_location"].exists():
            self.metadata["plot_save_location"].mkdir(parents=True)
        # raw_image_folder=Path(f"{self.raw_image_folder}/{self.metadata['dataset_name']}/mav0/cam0/data/")
        # gt_data = Path(f"/data/jagat/processed/evo/{metadata['dataset_name']}/{metadata['cam_type']}/lba/")
        
        # logging configuration
        if args.log_to_file:
            logging.basicConfig(
                filename=str(self.metadata["plot_save_location"].joinpath("app_2.log")),
                level=args.loglevel.upper(),
            )
        else:
            logging.basicConfig(level=args.loglevel.upper())

        logging.info(f"Near depth : {args.near_depth_threshold}")
        logging.info(f"Far depth : {args.far_depth_threshold}")
        logging.info(f"logging level = {logging.root.level}")
        logging.info(
            f"plots will be saved at : {self.metadata['plot_save_location'].absolute()}"
        )

    @property
    def metadata(self):
        return self.metadata
    
    @property
    def factor_graph_file(self) -> Path:
        if self.fg_file is not None:
            return self.fg_file
        else:
            raise RuntimeError("Filepath for processor not set")
    
    @factor_graph_file.setter
    def factor_graph_file(self, file_path: Path) -> None:
        self.fg_file = file_path
        self.metadata["cam_type"] = self.fg_file.parents[2].stem
        self.metadata["dataset_name"] = self.fg_file.parents[3].stem
        self.metadata["parent_folder"] = self.fg_file.parents[4]
        self.metadata["filename"] = self.fg_file.stem
        self.metadata["file_id"] = self.metadata["filename"].split("_")[1]
        rel_path = self.fg_file.relative_to(self.metadata["parent_folder"])
        folder_name = str(rel_path).replace("/", "_").split(".")[0]
        self.metadata["plot_save_location"] = Path(self.save_dir).joinpath(folder_name)
    
    
    @property
    def graph(self):
        if self.graph is not None:
            return self.graph
        else:
            raise Exception("Graph is not constructed")

    def get_covariance_analytics(self) -> Dict:
        return {
            "determinants": self.D_cat,
            "cov_singular_values": self.S_cat,
        }

    def covisibility_graph_to_adj_matrix(self) -> np.array:
        """
        builds an adjaceny matrix from the list of edges.
        returns an upper triangular matrix
        """
        src_nodes = self.ba_problem.src_nodes
        dst_nodes = self.ba_problem.dst_nodes
        node_ids = list(set(np.sort(src_nodes.numpy())))
        N = len(node_ids)
        M = np.zeros((N, N))
        for src, dst in zip(src_nodes, dst_nodes):
            for i, n in enumerate(node_ids):
                if src.item() == n:
                    row = i
                if dst.item() == n:
                    col = i
            M[row, col] = 1
        return np.triu(M)

    def initialize_ba_problem(self) -> Tuple:
        """
        builds a factor graph from the factor graph data.
        """

        fg_data = FactorGraphData.load_from_pickle_file(self.fg_file)
        self.ba_problem = BAProblem(fg_data)

        # set some parameters for solving.
        self.ba_problem.near_depth_threshold = self.near_depth_threshold
        self.ba_problem.far_depth_threshold = self.far_depth_threshold

        # set priors of the bundle adjustment problem
        oldest_nodes = self.ba_problem.get_oldest_poses_in_graph()
        prior_definition = self.ba_problem.set_prior_definition(
            pose_indices=oldest_nodes
        )
        logging.debug(f"Prior definition : {prior_definition}")

        # build a factor graph.
        self.ba_problem.add_visual_priors(prior_definition)
        if self.number_of_edges > 0:
            self.graph = self.ba_problem.build_visual_factor_graph(
                N_edges=self.number_of_edges
            )
        else:
            self.graph = self.ba_problem.build_visual_factor_graph(
                N_edges=self.ba_problem.edges
            )

        # store some more metadata for saving and plotting
        self.metadata["node_ids"] = self.ba_problem.get_node_ids(self.number_of_edges)
        self.metadata["image_ids"] = self.ba_problem.get_image_ids()
        logging.info(f"Prior node ids: {oldest_nodes}")
        logging.info(
            f"node ids - i: size: {len(self.metadata['node_ids'])} - {self.metadata['node_ids']}"
        )
        logging.info(
            f"node timestamps: size : {len(self.metadata['image_ids'])} - {self.metadata['image_ids']}"
        )

        return self.ba_problem, self.metadata

    def process(self, visualizer: None) -> bool:
        """
        computes covariance of the ba problem at init_vals
        and then compute its covariance
        return if true is there is no error
        """
        flag = False

        # Initialize the matrix for singular values and determinants.
        self.S_cat = np.zeros([len(self.metadata["node_ids"]), 6])
        self.D_cat = np.zeros([len(self.metadata["node_ids"]), 1])
        try:
            analyzer = GraphAnalysis(self.graph)
            marginals = analyzer.marginals(self.ba_problem.i_vals)
            symbols = lambda indices: [gtsam.symbol("x", idx) for idx in indices]
            cov_list = []
            for p_id in self.metadata["node_ids"]:
                cov = marginals.marginalCovariance(gtsam.symbol("x", p_id))
                cov_list.append(cov)
                U, S, V = np.linalg.svd(cov)
                self.S_cat[p_id, :] = S
                d = np.linalg.det(cov)
                self.D_cat[p_id, :] = d
                self._log_values(cov_list, p_id, cov, S, d)
            flag = True
        except Exception as e:
            logging.error(f"Filename - {self.metadata['filename']}, error code - {e}")
            time.sleep(2)
        return flag

    def _log_values(self, cov_list, p_id, cov, S, d):
        logging.debug(f" marginal covariance = {cov}")
        logging.debug(f" singular values at {p_id} = {S}")
        logging.debug(f" determinant at {p_id} = {d}")
        logging.debug(f"Concatenated singular values = {self.S_cat}")
        logging.debug(f"Concatenated covariance list : {cov_list}")

