import pytest
from factor_graph_insights.ba_problem import BAProblem, FactorGraphData
import torch
from pathlib import Path
import numpy as np
import gtsam
from numpy.testing import assert_allclose


@pytest.fixture
def keyframes():
    return 4


@pytest.fixture
def edges(keyframes):
    import math

    n_e = int(math.factorial(keyframes) / 2)
    return n_e


@pytest.fixture
def image_size():
    return (3, 3)


@pytest.fixture
def pkl_file_path():
    return Path("./tests/data/fg_data.pkl")


@pytest.fixture
def prior_noise_models():
    prior_rpy_sigma = 1
    # 3D translational standard deviation of of prior factor - gaussian model
    # (meters)
    prior_xyz_sigma = 0.05
    sigma_angle = np.deg2rad(prior_rpy_sigma)
    prior_noise_model_1 = gtsam.noiseModel.Diagonal.Sigmas(
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
    prior_noise_model_2 = gtsam.noiseModel.Diagonal.Sigmas(
        2
        * np.array(
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
    return prior_noise_model_1, prior_noise_model_2


@pytest.fixture
def factor_graph_data(keyframes, edges, image_size):
    # define poses for 4 keyframes - 4x7
    n_kf = keyframes
    n_e = edges
    ROWS, COLS = image_size
    poses = torch.zeros([n_kf, 7])  # tx, ty, tz, qx, qy, qz, qw

    # define disparity for each keyframe, here 3x3 is the image size
    disps = torch.zeros([n_kf, ROWS, COLS])
    # confidence weights with normal distribution for edges
    c_map = torch.randn([n_e, 2, ROWS, COLS])
    predicted = torch.zeros([n_e, 2, ROWS, COLS])
    # define the node i of i -j edge
    # unidirectional
    ii = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    jj = torch.tensor([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2])
    K = torch.tensor([50, 50, 50, 50])
    factor_graph_data = {
        "poses": poses,
        "disps": disps,
        "c_map": c_map,
        "graph_data": {
            "ii": ii,
            "jj": jj,
        },
        "predicted": predicted,
        "intrinsics": K,
    }
    return factor_graph_data


def test_BAProblem_constructor(factor_graph_data, image_size, keyframes, edges):
    n_kf = keyframes
    n_e = edges
    ba_problem = BAProblem(factor_graph_data)
    assert ba_problem.poses.shape == torch.Size([n_kf, 7])
    assert ba_problem.depth_maps.shape == torch.Size(
        [n_kf, image_size[0], image_size[1]]
    )
    assert ba_problem.confidence_map.shape == torch.Size(
        [n_e, 2, image_size[0], image_size[1]]
    )

    assert ba_problem.src_nodes.shape == torch.Size([n_e])
    assert ba_problem.dst_nodes.shape == torch.Size([n_e])

    assert ba_problem.calibration.shape == torch.Size([4])
    assert ba_problem.calibration_gtsam.shape == (5,)
    assert_allclose(ba_problem.calibration_gtsam, np.array([50, 50, 0, 50, 50]))
    assert ba_problem.predicted_pixels.shape == torch.Size(
        [n_e, 2, image_size[0], image_size[1]]
    )


def test_load_data_from_file(pkl_file_path):
    factor_graph_data = FactorGraphData.load_from_pickle_file(pkl_file_path)
    ba_problem = BAProblem(factor_graph_data)
    # check assumptions about shapes.
    n_kf = ba_problem.keyframes
    n_e = ba_problem.edges
    image_size = ba_problem.image_size
    assert image_size == (49, 61)
    assert ba_problem.poses.shape == torch.Size([n_kf, 7])
    assert ba_problem.confidence_map.shape == torch.Size(
        [n_e, 2, image_size[0], image_size[1]]
    )
    assert ba_problem.src_nodes.shape == torch.Size([n_e])
    assert ba_problem.calibration.shape == torch.Size([4])
    assert ba_problem.calibration_gtsam.shape == (5,)
    assert ba_problem.predicted_pixels.shape == torch.Size(
        [n_e, 2, image_size[0], image_size[1]]
    )


def test_build_graph_attributes(factor_graph_data):
    ba_problem = BAProblem(factor_graph_data)
    n_e = ba_problem.edges
    n_kf = ba_problem.keyframes
    image_size = ba_problem.image_size

    graph = ba_problem.build_visual_factor_graph(N_edges=n_e)

    expected_nrFactors = n_e * image_size[0] * image_size[1]
    assert graph.nrFactors() == expected_nrFactors
    keys = graph.keyVector()
    for i in range(0, n_kf):
        pose_key = gtsam.symbol("x", i)
        assert pose_key in keys, f"{pose_key} - x - {i} is absent in graph keys "
        counter = 0
        for row in range(0, image_size[0]):
            for col in range(0, image_size[1]):
                depth_key = gtsam.symbol(
                    "d", i * image_size[0] * image_size[1] + counter
                )
                assert (
                    depth_key in keys
                ), f"depth {row},{col} for image {i} is absent in graph keys"


def test_add_visual_priors(prior_noise_models, factor_graph_data):
    """
    test adding prior factors to factor graph
    """
    #
    prior_definition = {}
    ba_problem = BAProblem(factor_graph_data)

    with pytest.raises(AssertionError):
        ba_problem.add_visual_priors(prior_definition)

    prior_definition = {
        "prior_pose_symbols": (gtsam.symbol("x", 3), gtsam.symbol("x", 4)),
        "initial_poses": np.array(
            [[0, 0, 1, 0, 0, 0, 1], [0, 0.2, 1.5, -0.5, 0.0, 0.0, 0.86602]]
        ),
        "prior_noise_model": [prior_noise_models[0], prior_noise_models[1]],
    }
    ba_problem.add_visual_priors(prior_definition)
    graph = ba_problem.factor_graph
    assert graph.nrFactors() == len(prior_definition["prior_pose_symbols"])
    assert gtsam.symbol("x", 3) in graph.keyVector()
    assert gtsam.symbol("x", 4) in graph.keyVector()
    assert graph.at(0).noiseModel().equals(prior_noise_models[0], 1e-5)
    assert graph.at(1).noiseModel().equals(prior_noise_models[1], 1e-5)

    # prior factors after building factor graph
    ba_problem = BAProblem(factor_graph_data)
    n_e = ba_problem.edges
    n_kf = ba_problem.keyframes
    image_size = ba_problem.image_size

    ba_problem.add_visual_priors(prior_definition)
    graph = ba_problem.build_visual_factor_graph(N_edges=n_e)
    assert graph.nrFactors() == n_e * image_size[0] * image_size[1] + len(
        prior_definition["prior_pose_symbols"]
    )
