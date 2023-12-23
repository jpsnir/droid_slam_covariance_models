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
)


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
        required_keys = ["poses", "disps", "c_map", "ii", "jj"]

        assert (
            "poses" in factor_graph_data.keys()
        ), "Cannot initialize the BA problem. Dictionary Key 'poses' is missing"
        assert (
            "disps" in factor_graph_data.keys()
        ), "Cannot initialize the BA problem. Dictionary Key 'disps' is missing"
        assert (
            "c_map" in factor_graph_data.keys()
        ), "Cannot initialize the BA problem. Dictionary Key 'c_map' is missing"
        assert (
            "ii" in factor_graph_data.keys()
        ), "Cannot initialize the BA problem. Dictionary Key 'ii' is missing"
        assert (
            "jj" in factor_graph_data.keys()
        ), "Cannot initialize the BA problem. Dictionary Key 'jj' is missing"
        self._poses = factor_graph_data["poses"]
        self._disps = factor_graph_data["disps"]
        self._c_map = factor_graph_data["c_map"]
        self._ii = factor_graph_data["ii"]
        self._jj = factor_graph_data["jj"]


# def build_factor_graph(fg_data: dict, n: int = 0) -> gtsam.NonlinearFactorGraph:
#     """
#     build factor graph from complete data
#     """
#     graph_data = fg_data["graph_data"]
#     depth = fg_data["disps"]
#     poses = fg_data["poses"]
#     weights = fg_data["c_map"]
#     predicted = fg_data["predicted"]
#     K = fg_data["intrinsics"]
#     ii = graph_data["ii"]
#     jj = graph_data["jj"]
#     pair_unique_id = {}
#     if n == 0:
#         n = ii.size()[0]
#     unique_id = 0
#     full_graph = gtsam.NonlinearFactorGraph()
#     print(
#         f"""Graph index - {graph_data['ii'].size()},
#               poses - {poses.size()},
#               ---------------------------
#               weights - shape = {weights.size()},
#               ---------------------------
#               predicted - shape = {predicted.size()},
#               ---------------------------
#               depth - shape = {depth.size()},
#               -------------------------------
#               intrinics - shape = {K.size()}, {K},
#         """
#     )
#     for index, (ix, jx) in enumerate(zip(ii[:n], jj[:n])):
#         key = (ix, jx)
#         if key not in pair_unique_id.keys():
#             pair_unique_id[key] = unique_id
#             unique_id += 1
#         if max(ix, jx).item() > poses.size()[0] - 1:
#             print(f"Ignoring index - {ix , jx} - out of bounds")
#             continue
#         print(f"Index - {index} - Adding factors for {ix} - {jx} edge")
#         graph = factor_graph_image_pair(
#             i=ix,
#             j=jx,
#             pair_id=unique_id,
#             pose_i=poses[ix],
#             pose_j=poses[jx],
#             depth=depth[ix],
#             weights=weights[index],
#             target_pt=predicted[index],
#             intrinsics=K,
#         )
#         full_graph.push_back(graph)
#     print(f"Number of factors in full factor graph = {full_graph.nrFactors()}")
#     return full_graph
