import pytest
from factor_graph_insights.ba_problem import BAProblem
import torch


@pytest.fixture
def factor_graph_data():
    # define poses for 4 keyframes - 4x7
    n_kf = 4  # number of keyframes
    n_e = 6 * 2  # number of edges in covisibility graph.
    image_size = (3, 3)
    ROWS, COLS = image_size
    poses = torch.zeros([n_kf, 7])  # tx, ty, tz, qx, qy, qz, qw

    # define disparity for each keyframe, here 3x3 is the image size
    disps = torch.zeros([n_kf, ROWS, COLS])
    # confidence weights with normal distribution for edges
    c_map = torch.randn([2, n_e, ROWS, COLS])

    # define the node i of i -j edge
    # unidirectional
    ii = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    jj = torch.tensor([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2])
    factor_graph_data = {
        "poses": poses,
        "disps": disps,
        "c_map": c_map,
        "ii": ii,
        "jj": jj,
    }
    return factor_graph_data


def test_BAProblem_constructor(factor_graph_data):
    ba_problem = BAProblem(factor_graph_data)
