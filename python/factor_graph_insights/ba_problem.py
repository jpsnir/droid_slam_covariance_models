"""
forms the bundle adjustment problem from the complete visual data
of droid slam pickle file.
"""
from pathlib import Path
import pickle
from typing import (
    Union,
    List,
    Dict,
    Tuple,
)
import gtsam
import torch
import numpy as np
from factor_graph_insights.fg_builder import ImagePairFactorGraphBuilder
from factor_graph_insights.custom_factors.droid_error_functions import Droid_DBA_Error
from factor_graph_insights.fg_builder import DataConverter


class FactorGraphData:
    @staticmethod
    def load_from_pickle_file(filename: Union[str, Path]) -> Dict:
        """
        generates parsed data from pickle file
        """
        fg_data = {}
        with open(filename, "rb") as f:
            fg_data = pickle.load(f)

        return fg_data

    @staticmethod
    def log_factor_graph_stats(fg_data: dict):
        """
        prints some basic details of the factor graph for
        error checking
        """
        print(f"data keys: {fg_data.keys()}")
        print(f"id: {fg_data['id']}")
        print(f"intrinsics: {fg_data['intrinsics']}")
        print("------------------------------")
        print(f"graph data - ii: {fg_data['graph_data']['ii']}")
        print(f"graph data - jj: {fg_data['graph_data']['jj']}")
        print(
            f"graph data - number of connections:\
            {fg_data['graph_data']['ii'].size().numel()}"
        )
        print(f"weights size: {fg_data['c_map'].size()}")
        print(f"target pts size: {fg_data['predicted'].size()}")
        print("------------------------------")
        print(f"tstamps size: {fg_data['tstamp'].size()}")
        print(f"poses size: {fg_data['poses'].size()}")
        print(f"disparity size: {fg_data['disps'].size()}")


class BAProblem:
    """
    builds a BA problem from factor graph data given the data.
    """

    def __init__(self, factor_graph_data: dict):
        required_keys = [
            "poses",
            "disps",
            "c_map",
            "graph_data",
            "intrinsics",
            "predicted",
        ]

        for r_key in required_keys:
            assert (
                r_key in factor_graph_data.keys()
            ), f"Cannot initialize the BA problem. Dictionary Key {r_key} is missing"
        # check keys
        assert (
            "ii" in factor_graph_data["graph_data"].keys()
        ), "Cannot initialize BA problem, src (ii) nodes data does not exist in graph data"
        assert (
            "jj" in factor_graph_data["graph_data"].keys()
        ), "Cannot initialize BA problem, dst (jj) nodes data does not exist in graph data"
        assert (
            factor_graph_data["graph_data"]["ii"].shape
            == factor_graph_data["graph_data"]["jj"].shape
        ), "Cannot initialize BA problem, Covisibility graph size mismatch, src(i) nodes not equal to dst(j) nodes"

        self._poses = factor_graph_data["poses"]
        self._depths = factor_graph_data["disps"]
        self._c_map = factor_graph_data["c_map"]
        self._ii = factor_graph_data["graph_data"]["ii"]
        self._jj = factor_graph_data["graph_data"]["jj"]
        self._K = factor_graph_data["intrinsics"]
        self._convert_to_gtsam_K()
        self._predicted = factor_graph_data["predicted"]
        self._init_values = gtsam.Values()
        self._graph = None
        self._prior_added = False

    @property
    def keyframes(self) -> int:
        n_kf, _ = self._poses.shape
        return n_kf

    @property
    def edges(self) -> int:
        n_e = self._ii.shape
        return n_e[0]

    @property
    def image_size(self) -> Tuple[int, int]:
        n, ROWS, COLS = self._depths.shape
        return (ROWS, COLS)

    @property
    def poses(self) -> torch.Tensor:
        return self._poses

    @property
    def depth_maps(self) -> torch.Tensor:
        return self._depths

    @property
    def confidence_map(self) -> torch.Tensor:
        return self._c_map

    @property
    def src_nodes(self) -> torch.Tensor:
        return self._ii

    @property
    def dst_nodes(self) -> torch.Tensor:
        return self._jj

    @property
    def prior_flag(self) -> bool:
        return self._prior_added

    @property
    def calibration(self) -> torch.Tensor:
        return self._K

    @property
    def calibration_gtsam(self) -> np.ndarray:
        return self._gtsam_kvec

    @property
    def predicted_pixels(self) -> np.ndarray:
        return self._predicted

    def get_oldest_poses_in_graph(self, n: int = 2) -> List:
        nodes_i = list(np.sort(self.src_nodes.numpy()))
        nodes_i_unique = list(set(nodes_i))
        return nodes_i_unique[:n]

    @property
    def factor_graph(self) -> gtsam.NonlinearFactorGraph:
        assert (
            self._graph is not None
        ), " Graph is not yet constructed, build it from the data first"
        return self._graph

    def _convert_to_gtsam_K(self):
        k = self._K.numpy()
        self._gtsam_kvec = np.array([k[0], k[1], 0, k[2], k[3]])

    @property
    def i_vals(self) -> gtsam.Values:
        return self._init_values

    def _add_pose_priors(
        self,
        graph: gtsam.NonlinearFactorGraph,
        symbols: List,
        prior_poses: np.ndarray,
        prior_noise_models: List,
    ) -> gtsam.NonlinearFactorGraph:
        """
        prior_poses world to camera frame
        """

        for i in range(0, len(symbols)):
            symbol = symbols[i]
            prior_noise_model = prior_noise_models[i]
            gtsam_pose = DataConverter.to_gtsam_pose(prior_poses[i])
            graph.addPriorPose3(symbol, gtsam_pose, prior_noise_model)

    def set_prior_noise_model(
        self, prior_rpy_sigma: float = 1, prior_xyz_sigma: float = 0.05
    ) -> gtsam.noiseModel:
        """_summary_

        Args:
            prior_xyz_sigma (float, optional): 3D rotational standard deviation of prior factor - gaussian model  (meters)
            prior_rpy_sigma (float, optional): - 3D translational standard deviation of of prior factor - gaussian model (degrees)

        Returns:
            List[gtsam.noiseModel]: _description_
        """
        sigma_angle = np.deg2rad(prior_rpy_sigma)
        prior_noise_model = gtsam.noiseModel.Diagonal.Sigmas(
            np.array(
                [
                    sigma_angle,
                    sigma_angle,
                    sigma_angle,
                    prior_xyz_sigma,
                    prior_xyz_sigma,
                    prior_xyz_sigma,
                ]
            )
        )

        return prior_noise_model

    def set_prior_definition(
        self,
        pose_indices: List,
        prior_noise_models: List = None,
    ) -> Dict:
        """creates a dictionary for defining prior poses in the factor graph"""
        assert (
            len(pose_indices) >= 2
        ), " For visual factor graph at least two priors are required"

        # lambda function to create symbols from indices
        symbols = lambda indices: [gtsam.symbol("x", idx) for idx in indices]

        # set the default prior noise model if not provided
        if prior_noise_models is None:
            prior_noise_models = [self.set_prior_noise_model()] * len(pose_indices)

        p_poses_c_w = np.zeros((len(pose_indices), 7))
        for i, index in enumerate(pose_indices):
            # assert (
            #     self.poses[index] == torch.tensor([0, 0, 0, 0, 0, 0, 1])
            # ).all(), f"{index} - {self.poses[index]}"
            p_poses_c_w[i, :] = DataConverter.invert_pose(self.poses[index])

        prior_definition = {
            "prior_pose_symbols": symbols(pose_indices),
            "initial_poses": p_poses_c_w,
            "prior_noise_model": prior_noise_models,
        }
        return prior_definition

    def add_visual_priors(self, priors_definition: Dict) -> gtsam.NonlinearFactorGraph:
        """prior factor in the graph"""
        # check inputs
        # TODO: Decorate this type checking code.
        assert (
            "prior_pose_symbols" in priors_definition
        ), " Required key missing, define 'prior_pose_symbols' in dict"
        assert (
            "initial_poses" in priors_definition
        ), "Required key missing, define 'initial_poses' key in dict"
        assert (
            "prior_noise_model" in priors_definition
        ), "Required Key missing, define 'prior_noise_model' key in the dict"
        assert isinstance(
            priors_definition["prior_pose_symbols"], (tuple, list)
        ), " 'prior_pose_symbols must be a tuple of ints"
        assert isinstance(
            priors_definition["initial_poses"], np.ndarray
        ), " 'initial_poses' must be nx7 numpy arrays, world to camera frame, the order being: tx, ty, tz, qx, qy, qz, qw"
        assert isinstance(
            priors_definition["prior_noise_model"], (list, tuple)
        ), " 'prior_noise_model' must be a list of gtsam.noiseModel for Pose3"
        assert (
            priors_definition["initial_poses"].shape[1] == 7
        ), " 'initial_poses' must be nx7"
        assert (
            len(priors_definition["prior_pose_symbols"])
            == priors_definition["initial_poses"].shape[0]
        ), " Number of symbols not equal to number of poses"
        assert len(priors_definition["prior_noise_model"]) == len(
            priors_definition["prior_pose_symbols"]
        ), " Number of noise models not equal to number of pose symbols"

        if self._graph is None:
            self._graph = gtsam.NonlinearFactorGraph()

        prior_n_models = priors_definition["prior_noise_model"]
        symbols = priors_definition["prior_pose_symbols"]
        p_poses_c_w = priors_definition["initial_poses"]

        self._add_pose_priors(
            graph=self._graph,
            symbols=symbols,
            prior_noise_models=prior_n_models,
            prior_poses=p_poses_c_w,
        )
        self._prior_added = True

    def build_visual_factor_graph(self, N_edges=5) -> gtsam.NonlinearFactorGraph:
        """
        builds a factor graph from complete factor graph data
        N_prior : poses that will be assigned prior, default 2
        """
        assert (
            self.prior_flag
        ), "Priors of poses are not set, System may become under-determined. Set them first"
        image_size = self.image_size
        if self._graph is None:
            self._graph = gtsam.NonlinearFactorGraph()
        
        for edge_id, (node_i, node_j) in enumerate(
            zip(self._ii[:N_edges], self._jj[:N_edges])
        ):
            pose_cam_i_w = self._poses[node_i]
            pose_cam_j_w = self._poses[node_j]

            # handle stereo case
            if node_i == node_j:
                pose_ji = torch.tensor([0, 0, -0.11, 0, 0, 0, 1])
                pose_ji_gtsam = DataConverter.to_gtsam_pose(pose_ji)
                pose_cam_i_w_gtsam = DataConverter.to_gtsam_pose(pose_w_cam_i)
                pose_cam_j_w_gtsam = gtsam.Pose3(pose_ji_gtsam.matrix()@pose_cam_i_w_gtsam.matrix())
                pose_cam_j_w = DataConverter.to_pose_with_quaternion(pose_cam_j_w_gtsam)
            
            pose_w_cam_i = DataConverter.invert_pose(pose_cam_i_w)
            pose_w_cam_j = DataConverter.invert_pose(pose_cam_j_w)
            fg_builder = (
                ImagePairFactorGraphBuilder(node_i, node_j, image_size)
                .set_calibration(self.calibration_gtsam)
                .set_depths(self._depths[node_i])
                .set_poses_and_cameras(pose_w_cam_i, pose_w_cam_j)
                .set_pixel_weights(self._c_map[edge_id])
                .set_target_pts(self._predicted[edge_id])
                .set_error_model(Droid_DBA_Error(self._gtsam_kvec))
            )
            graph = fg_builder.build_factor_graph(self.i_vals)
            updated_i_vals = fg_builder.init_values_image_pair
            self._init_values = updated_i_vals
            self._graph.push_back(graph)
        return self._graph

    # TODO: Incremental bundle adjustment : how can we do it, what is the theory behind it? Simplest example
    #  to test it and build the incremental
    def build_incremental_visual_factor_graph(self):
        """"""
