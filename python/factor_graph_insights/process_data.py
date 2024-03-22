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


class FactorGraphFileProcessor:
    """
    """
    def __init__(self, args):
        """
        initializes the factor graph processer with input arguments
        """
        
        self.near_depth_threshold = args.near_depth_threshold
        self.far_depth_threshold = args.far_depth_threshold
        self.number_of_edges = args.number_of_edges
        if self.fg_dir.exists():
            self.files_list = sorted(os.listdir(self.fg_dir))
        logging.info(f"Number of files to processed in {self.fg_dir} = {len(self.files_list[start:end])}")
        self.metadata = {}
        self.D_cat = None
        self.S_cat = None
    
    def get_covariance_analytics(self) -> Dict:
        return {
            "determinants": self.D_cat,
            "cov_singular_values": self.S_cat,
        }
    
    def process(self, fg_file: Path) -> bool:
        """
        formulate the factorgraph file as a ba problem 
        and then compute its covariance
        return if true is there is no error
        """
        flag = False
        filename = fg_file.stem
        fg_data = FactorGraphData.load_from_pickle_file(fg_file)
        ba_problem = BAProblem(fg_data)
        ba_problem.near_depth_threshold = self.near_depth_threshold
        ba_problem.far_depth_threshold = self.far_depth_threshold
        
        oldest_nodes = ba_problem.get_oldest_poses_in_graph()
        prior_definition = ba_problem.set_prior_definition(pose_indices=oldest_nodes)
        logging.debug(f"Prior definition : {prior_definition}")
        ba_problem.add_visual_priors(prior_definition)
        if self.number_of_edges > 0:
            graph = ba_problem.build_visual_factor_graph(N_edges=self.number_of_edges)
        else:
            graph = ba_problem.build_visual_factor_graph(N_edges=ba_problem.edges)
        init_vals = ba_problem.i_vals
        self.S_cat = np.zeros([len(ba_problem.poses), 6])
        self.D_cat = np.zeros([len(ba_problem.poses), 1])
        node_ids_i = ba_problem.get_node_ids(self.number_of_edges)
        image_ids = ba_problem.get_image_ids()
        logging.info(f"Prior node ids: {oldest_nodes}")
        logging.info(f"node ids - i: size: {len(node_ids_i)} - {node_ids_i}")
        logging.info(f"node timestamps: size : {len(image_ids)} - {image_ids}")

        try:
            analyzer = GraphAnalysis(graph)
            marginals = analyzer.marginals(init_vals)
            symbols = lambda indices: [gtsam.symbol("x", idx) for idx in indices]
            cov_list = []
            for p_id in node_ids_i:
                cov = marginals.marginalCovariance(gtsam.symbol("x", p_id))
                cov_list.append(cov)
                U, S, V = np.linalg.svd(cov)
                self.S_cat[p_id, :] = S
                d = np.linalg.det(cov)
                self.D_cat[p_id, :] = d
                logging.debug(f" marginal covariance = {cov}")
                logging.debug(f" singular values at {p_id} = {S}")
                logging.debug(f" determinant at {p_id} = {d}")
                logging.debug(f"Concatenated singular values = {self.S_cat}")
                logging.debug(f"Concatenated covariance list : {cov_list}")
            flag = True
        except Exception as e:
            logging.error(f"Filename - {filename}, error code - {e}")
            time.sleep(2)
        return flag

class FgFolderProcessor(FactorGraphFileProcessor):
    
    def __init__(self, args):
        super().__init__(args=args)
        self.fg_dir = Path(args.dir)
        self.plot_folder_name = self._set_paths(self.fg_dir, parent_level=3)    
        self.plot_save_location = Path(args.save_dir).joinpath(self.plot_folder_name)
        if not self.plot_save_location.exists(): 
            self.plot_save_location.mkdir(parents=True)
        
        # setup logging configuration
        logging.basicConfig(
            filename=str(self.plot_save_location.joinpath("app.log")),
            level=args.loglevel.upper()
        )
        logging.info(f"logging level = {logging.root.level}")
        logging.info(f"plots will be saved at : {self.plot_save_location.absolute()}")
    
        self.start = args.start_id
        self.end = args.end_id
        self.files_that_worked = []
        self.files_that_failed = []
        
    
    def _set_paths(self, fg_dir: Path, parent_level:int = 3) -> Path:
        """
        sets relatives path and plot folder name
        path name
        """
        parent_path = self.fg_dir.parents[parent_level]
        self.rel_path = self.fg_dir.relative_to(parent_path)
        plot_folder_name = str(self.rel_path).replace("/","_")
        return plot_folder_name
    
    def process(self):
        """
        analyzing all factor graph files within the start and  end range.
        """
        for file_num, filename in enumerate(self.files_list[self.start: self.end]):
            logging.info(f"Files worked : {self.start + file_num}")
            fg_file = self.fg_dir.joinpath(filename)
            # process individual file
            flag = super().process(fg_file)
            if flag:
                self.files_that_worked.append(int(filename.split("_")[1]))
            else:
                self.files_that_failed.append(int(filename.split("_")[1]))
        logging.info(f"files that worked : {self.files_that_worked}")
        logging.info(f"files that failed : {self.files_that_failed}")
