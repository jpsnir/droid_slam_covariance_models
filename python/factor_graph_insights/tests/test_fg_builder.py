import pytest
import os
from functools import partial
import numpy as np
import torch
import gtsam
from numpy.testing import assert_allclose
from factor_graph_insights.custom_factors import droid_error_functions as droid_ef
import scipy.spatial.transform.rotation as rotation
from factor_graph_insights.fg_builder import ImagePairFactorGraphBuilder
from factor_graph_insights.fg_builder import DataConverter
from factor_graph_insights.custom_factors.droid_error_functions import Droid_DBA_Error


@pytest.fixture
def input_data():
    pixel_coords = [gtsam.Point2(10, 60), gtsam.Point2(50, 50), gtsam.Point2(30, 40)]
    errors_expected = [gtsam.Point2(5, 5), gtsam.Point2(12, -10), gtsam.Point2(23, 32)]
    depths = torch.tensor([1, 1, 1]).reshape(1, 3)
    weights = torch.tensor([[[0.9, 0.8, 0.8]], [[0.8, 0.6, 0.7]]])
    target_pt = torch.tensor([[18.3333, 38, 13.6667], [51.6667, 60, 11.3333]]).reshape(
        2, 1, 3
    )
    intrinsics = torch.tensor([50, 50, 0, 50, 50])
    quat_w_cam = gtsam.Rot3.RzRyRx(-np.pi / 2, 0, -np.pi / 2).toQuaternion().coeffs()
    p_w_cam_i = np.array([2, 1, 0])
    p_w_cam_j = np.array([1.5, 0, 0])
    p_i = torch.tensor(np.concatenate((p_w_cam_i, quat_w_cam)))
    p_j = torch.tensor(np.concatenate((p_w_cam_j, quat_w_cam)))
    gtsam_pose_i = DataConverter.to_gtsam_pose(p_i.numpy())
    gtsam_pose_j = DataConverter.to_gtsam_pose(p_j.numpy())
    camera = gtsam.Cal3_S2(fx=50, fy=50, s=0, u0=50, v0=50)
    ph_camera_i = gtsam.PinholeCameraCal3_S2(pose=gtsam_pose_i, K=camera)
    ph_camera_j = gtsam.PinholeCameraCal3_S2(pose=gtsam_pose_j, K=camera)
    return dict(
        [
            ("pixels_i", pixel_coords),
            ("depths", depths),
            ("weights", weights),
            ("target_pt", target_pt),
            ("intrinsics", intrinsics),
            ("pose_i", p_i),
            ("pose_j", p_j),
            ("gtsam_pose_i", gtsam_pose_i),
            ("gtsam_pose_j", gtsam_pose_j),
            ("camera", camera),
            ("ph_camera_i", ph_camera_i),
            ("ph_camera_j", ph_camera_j),
        ]
    )


@pytest.fixture
def fg_builder(input_data):
    fg_builder = ImagePairFactorGraphBuilder(0, 1, (1, 3))
    fg_builder.set_calibration(input_data["intrinsics"]).set_inverse_depths(
        input_data["depths"]
    ).set_pixel_weights(input_data["weights"]).set_target_pts(
        input_data["target_pt"]
    ).set_poses_and_cameras(
        pose_i=input_data["pose_i"], pose_j=input_data["pose_j"]
    )
    k_vec = fg_builder.calibration.numpy()
    dba_error = Droid_DBA_Error(k_vec)
    fg_builder.error_model = dba_error

    return fg_builder


def test_gtsam_transform_funcs():
    pt_w_ = np.array([1, 0, 0])
    pt_1_ = np.array([1, 0, 2])
    pose_w_1 = gtsam.Pose3(
        np.array([[1, 0, 0, 1.5], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    )
    # transform to converts point in base frame to current frame
    pt_1 = pose_w_1.transformTo(pt_w_)
    # transform from converts point in current frame to world frame.
    pt_w = pose_w_1.transformFrom(pt_1_)

    assert_allclose(pt_1, np.array([-0.5, 0, 0]))
    assert_allclose(pt_w, np.array([2.5, 0, 2]))


def test_data_converter():
    r = gtsam.Rot3.Identity()
    q = r.toQuaternion().coeffs()  # qx, qy, qz, qw
    p = np.array([0, 0, 1])  # tx, ty, tz
    pose = np.concatenate((p, q))
    pose_gtsam = DataConverter.to_gtsam_pose(pose)
    assert_allclose(p, pose_gtsam.translation())
    assert_allclose(q, pose_gtsam.rotation().toQuaternion().coeffs())

    pose_inverse_expected = np.concatenate((-p, q))
    pose_inverse_computed = DataConverter.invert_pose(pose)
    assert_allclose(pose_inverse_computed, pose_inverse_expected)


def test_factor_graph_builder_construction(fg_builder, input_data):
    """Test the factor graph builder class"""

    assert fg_builder.dst_img_id == 1
    assert fg_builder.src_img_id == 0
    assert fg_builder.image_size == (1, 3)
    assert fg_builder.pixel_weights.shape == torch.Size([2, 1, 3])
    assert fg_builder.target_pts.shape == torch.Size([2, 1, 3])
    assert fg_builder.depths.shape == torch.Size([1, 3])
    assert torch.equal(fg_builder.pixel_weights, input_data["weights"])
    assert torch.equal(fg_builder.calibration, input_data["intrinsics"])
    assert torch.equal(fg_builder.depths, input_data["depths"])

    assert torch.equal(fg_builder.poses[0], input_data["pose_i"])
    assert torch.equal(fg_builder.poses[1], input_data["pose_j"])
    assert_allclose(
        fg_builder.poses[0][:3].numpy(), input_data["gtsam_pose_i"].translation()
    )
    q_gtsam = input_data["gtsam_pose_i"].rotation().toQuaternion()
    q = np.array([q_gtsam.w(), q_gtsam.x(), q_gtsam.y(), q_gtsam.z()])
    assert_allclose(fg_builder.poses[0][3:].numpy(), q)

    assert torch.equal(fg_builder.poses[1], input_data["pose_j"])
    assert fg_builder.camera.equals(input_data["camera"], 1e-5)
    assert fg_builder.pinhole_cameras[0].equals(input_data["ph_camera_i"], 1e-5)
    assert fg_builder.pinhole_cameras[1].equals(input_data["ph_camera_j"], 1e-5)


def test_factor_graph_builder_point_depth_check(fg_builder, input_data):
    """error evaluation function of custom factor"""
    k_vec = np.array([50, 50, 0, 50, 50])
    dba_error = Droid_DBA_Error(k_vec)
    dba_error.pixel_to_project = input_data["pixels_i"][0]
    dba_error.predicted_pixel = np.array([18.3333, 51.6667])
    gtsam_p_i = input_data["gtsam_pose_i"]
    gtsam_p_j = input_data["gtsam_pose_j"]
    depths = input_data["depths"]
    fg_builder.error_model = dba_error
    pj = input_data["pose_j"]
    p_i = torch.tensor([0, 0, 0, 0, 0, 0, 1])
    p_j_1 = torch.tensor([0, 0, 0.5, 0, 0, 0, 1])
    p_j_2 = torch.tensor([0, 0, 0.8, 0, 0, 0, 1])

    fg_builder.set_poses_and_cameras(
        pose_i=p_i,
        pose_j=p_j_1,
    )
    depth_j, flag = fg_builder.depth_to_cam_j(input_data["pixels_i"][0], 1, 0.25)
    assert depth_j == 0.5

    fg_builder.set_poses_and_cameras(
        pose_i=p_i,
        pose_j=p_j_2,
    )
    depth_j, (flag_i, flag_j) = fg_builder.depth_to_cam_j(
        input_data["pixels_i"][0], 1, 0.25
    )
    assert_allclose(depth_j, 0.2)
    assert flag_j == True


def test_build_factor_graph(fg_builder, input_data):
    """_summary_"""
    assert fg_builder._image_size == torch.Size((1, 3))
    ROWS, COLS = fg_builder._image_size
    assert ROWS == 1
    assert COLS == 3
    graph = fg_builder.build_factor_graph()
    # check number of factors
    assert graph.nrFactors() == 3
    assert graph.exists(0)
    assert graph.exists(1)
    assert graph.exists(2)
    assert not graph.exists(3)
    assert graph.size() == 3


def test_graph_symbols(fg_builder, input_data):
    # check symbols
    graph = fg_builder.build_factor_graph()
    x_i = gtsam.symbol("x", fg_builder.i)
    x_j = gtsam.symbol("x", fg_builder.j)
    d_i = gtsam.symbol("d", 0)

    factor = graph.at(0)
    assert factor.keys() == [x_i, x_j, d_i]
    factor = graph.at(1)
    assert factor.keys() == [x_i, x_j, d_i + 1]
    factor = graph.at(2)
    assert factor.keys() == [x_i, x_j, d_i + 2]

    fg_builder.src_img_id = 1
    fg_builder.dst_img_id = 2
    graph = fg_builder.build_factor_graph()

    x_i = gtsam.symbol("x", 1)
    x_j = gtsam.symbol("x", 2)
    d_i = gtsam.symbol("d", 1 * 1 * 3)

    factor = graph.at(0)
    assert factor.keys() == [x_i, x_j, d_i]
    factor = graph.at(1)
    assert factor.keys() == [x_i, x_j, d_i + 1]
    factor = graph.at(2)
    assert factor.keys() == [x_i, x_j, d_i + 2]

    # arbitrary src image
    fg_builder.src_img_id = 10
    fg_builder.dst_img_id = 15
    graph = fg_builder.build_factor_graph()

    x_i = gtsam.symbol("x", 10)
    x_j = gtsam.symbol("x", 15)
    d_i = gtsam.symbol("d", 10 * 1 * 3)

    factor = graph.at(0)
    assert factor.keys() == [x_i, x_j, d_i]
    factor = graph.at(1)
    assert factor.keys() == [x_i, x_j, d_i + 1]
    factor = graph.at(2)
    assert factor.keys() == [x_i, x_j, d_i + 2]
    # check graph


def test_graph_factors(fg_builder):
    graph = fg_builder.build_factor_graph()
    pose_i, pose_j = fg_builder.poses
    pose_gtsam_i = DataConverter.to_gtsam_pose(pose_i)
    pose_gtsam_j = DataConverter.to_gtsam_pose(pose_j)
    x_i = gtsam.symbol("x", fg_builder.i)
    x_j = gtsam.symbol("x", fg_builder.j)
    d_i = gtsam.symbol("d", 0)

    v = gtsam.Values()
    v.insert(x_i, pose_gtsam_i)
    v.insert(x_j, pose_gtsam_j)
    v.insert(d_i, fg_builder.depthAt([0, 0]))
    v.insert(d_i + 1, fg_builder.depthAt([0, 1]))
    v.insert(d_i + 2, fg_builder.depthAt([0, 2]))
    unwhitened_error = (graph.at(0)).unwhitenedError(v)
    k_vec = fg_builder.calibration.numpy()
    dba_error = Droid_DBA_Error(k_vec)
    dba_error.pixel_to_project = np.array([0, 0])
    dba_error.predicted_pixel = fg_builder.target_pts[:, 0, 0].numpy()
    actual_error, J = dba_error.error(
        pose_gtsam_i, pose_gtsam_j, fg_builder.depthAt([0, 0])
    )
    assert_allclose(actual_error, unwhitened_error.reshape(2, 1))

    ## Jacobian
    dba_error.pixel_to_project = np.array([0, 1])
    dba_error.predicted_pixel = fg_builder.target_pts[:, 0, 1].numpy()

    depth_j, (is_close_i, is_close_j) = fg_builder.depth_to_cam_j((0, 1), 0.25)
    assert is_close_j == False
    assert is_close_i == False
    actual_error, J = dba_error.error(
        pose_gtsam_i, pose_gtsam_j, fg_builder.depthAt([0, 1])
    )
    unwhitened_error = (graph.at(1)).unwhitenedError(v)
    assert_allclose(actual_error, unwhitened_error.reshape(2, 1))
    jac_factor = (graph.at(1)).linearize(v)
    H1 = jac_factor.jacobian()[0][:, 0:6]
    H2 = jac_factor.jacobian()[0][:, 6:12]
    H3 = jac_factor.jacobian()[0][:, 12].reshape(2, 1)
    confidence_factor = 0.001
    confidence = confidence_factor * fg_builder.pixel_weights[:, 0, 1].numpy()
    assert confidence.shape == (2,)
    I = np.diag(confidence)
    assert H1.shape == (2, 6)
    assert J[0].shape == (2, 6)
    assert_allclose(H1, np.sqrt(I) @ J[0])
    assert_allclose(H2, np.sqrt(I) @ J[1])
    assert_allclose(H3, np.sqrt(I) @ J[2])
