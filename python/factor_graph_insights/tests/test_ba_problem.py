import pytest
from factor_graph_insights.ba_problem import BAProblem
import torch


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
def factor_graph_data(keyframes, edges, image_size):
    # define poses for 4 keyframes - 4x7
    n_kf = keyframes
    n_e = edges
    ROWS, COLS = image_size
    poses = torch.zeros([n_kf, 7])  # tx, ty, tz, qx, qy, qz, qw

    # define disparity for each keyframe, here 3x3 is the image size
    disps = torch.zeros([n_kf, ROWS, COLS])
    # confidence weights with normal distribution for edges
    c_map = torch.randn([2, n_e, ROWS, COLS])
    predicted = torch.zeros([n_e, ROWS, COLS, 2])
    # define the node i of i -j edge
    # unidirectional
    ii = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    jj = torch.tensor([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2])
    K = torch.tensor([50, 50, 50, 50])
    factor_graph_data = {
        "poses": poses,
        "disps": disps,
        "c_map": c_map,
        "ii": ii,
        "jj": jj,
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
        [2, n_e, image_size[0], image_size[1]]
    )

    assert ba_problem.src_nodes.shape == torch.Size([n_e])
    assert ba_problem.dst_nodes.shape == torch.Size([n_e])

    assert ba_problem.edge_node_i
