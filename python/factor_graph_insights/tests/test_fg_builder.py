import pytest
import os
from functools import partial
import numpy as np
import torch
import gtsam
from factor_graph_insights.custom_factors import droid_error_functions as droid_ef
import scipy.spatial.transform.rotation as rotation
from factor_graph_insights.fg_builder import ImagePairFactorGraphBuilder
from factor_graph_insights.fg_builder import DataConverter
from factor_graph_insights.custom_factors.droid_error_functions import Droid_DBA_Error


@pytest.fixture
def input_data():
    depths = torch.tensor([1, 1, 1])
    weights = torch.tensor([[0.9, 0.8], [0.8, 0.6], [0.8, 0.7]])
    target_pt = torch.tensor([[35, 45], [56, 66], [44, 54]])
    intrinsics = torch.tensor([50, 50, 50, 50])
    r_w_cam = rotation.Rotation.from_euler("zyx", [-np.pi / 2, 0, -np.pi / 2])
    p_w_cam_i = np.array([0, 0, 0])
    p_w_cam_j = np.array([1, 0, 0])
    quat_w_cam = r_w_cam.as_quat()
    p_i = torch.tensor(np.concatenate((p_w_cam_i, quat_w_cam)))
    p_j = torch.tensor(np.concatenate((p_w_cam_j, quat_w_cam)))
    gtsam_pose_i = DataConverter.to_gtsam_pose(p_i.numpy())
    gtsam_pose_j = DataConverter.to_gtsam_pose(p_j.numpy())
    camera = gtsam.Cal3_S2(fx=50, fy=50, s=0, u0=50, v0=50)
    ph_camera_i = gtsam.PinholeCameraCal3_S2(pose=gtsam_pose_i, K=camera)
    ph_camera_j = gtsam.PinholeCameraCal3_S2(pose=gtsam_pose_j, K=camera)
    return dict(
        [
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
    fg_builder = ImagePairFactorGraphBuilder(0, 1, (3, 1))
    fg_builder.set_calibration(input_data["intrinsics"]).set_depths(
        input_data["depths"]
    ).set_pixel_weights(input_data["weights"]).set_target_pts(
        input_data["target_pt"]
    ).set_poses(
        pose_i=input_data["pose_i"], pose_j=input_data["pose_j"]
    ).create_pinhole_cameras()
    return fg_builder


def test_factor_graph_builder_construction(fg_builder, input_data):
    """Test the factor graph builder class"""

    assert torch.equal(fg_builder.pixel_weights, input_data["weights"])
    assert torch.equal(fg_builder.calibration, input_data["intrinsics"])
    assert torch.equal(fg_builder.depths, input_data["depths"])
    assert fg_builder.dst_img_id == 1
    assert fg_builder.src_img_id == 0
    assert fg_builder.image_size == (3, 1)
    assert torch.equal(fg_builder.poses[0], input_data["pose_i"])
    assert torch.equal(fg_builder.poses[1], input_data["pose_j"])
    assert DataConverter.to_gtsam_pose(fg_builder.poses[0]).equals(
        input_data["gtsam_pose_i"], 1e-5
    )
    assert torch.equal(fg_builder.poses[1], input_data["pose_j"])
    assert fg_builder.camera.equals(input_data["camera"], 1e-5)
    assert fg_builder.pinhole_cameras[0].equals(input_data["ph_camera_i"], 1e-5)
    assert fg_builder.pinhole_cameras[1].equals(input_data["ph_camera_j"], 1e-5)


def test_factor_graph_builder_custom_factor(fg_builder, input_data):
    """error evaluation function of custom factor"""

    k_vec = np.array([50, 50, 0, 50, 50])
    dba_error = Droid_DBA_Error(k_vec)
    dba_error.pixel_to_project = np.array([10, 60])
    dba_error.predicted_pixel = np.array([18.3333, 51.6667])
    gtsam_p_i = input_data["gtsam_pose_i"]
    gtsam_p_j = input_data["gtsam_pose_j"]
    depths = input_data["depths"]
    fg_builder.set_custom_factor_residual(dba_error)


def test_factor_graph_image_pair():
    """test the factor graph generated from this function"""
    i = 1
    j = 2
    # orientation of camera wrt world
    r_w_cam = rotation.Rotation.from_euler("zyx", [-np.pi / 2, 0, -np.pi / 2])
    p_w_cam_i = np.array([0, 0, 0])
    p_w_cam_j = np.array([1, 0, 0])
    quat_w_cam = r_w_cam.as_quat()

    pose_i = torch.tensor(np.concatenate((p_w_cam_i, quat_w_cam)))
    pose_j = torch.tensor(np.concatenate((p_w_cam_j, quat_w_cam)))
