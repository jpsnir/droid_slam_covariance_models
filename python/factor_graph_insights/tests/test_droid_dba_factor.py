""" Replicate tests written down in c++ to check whether the 
function implemented has the same validity.
"""
import pytest
from factor_graph_insights.custom_factors.droid_error_functions import Droid_DBA_Error
import numpy as np
from numpy.testing import assert_allclose
import gtsam


def test_constructor():
    kvec = np.array([50.0, 50.0, 0.0, 50.0, 50.0])
    dba_error = Droid_DBA_Error(kvec)
    dba_error.predicted_pixel = np.array([10, 10]).reshape(2, 1)
    dba_error.pixel_to_project = np.array([4, 5]).reshape(2, 1)

    assert (dba_error.predicted_pixel == np.array([10, 10]).reshape(2, 1)).all()
    assert (dba_error.calibration.vector() == kvec).all()
    assert (dba_error.pixel_to_project == np.array([4, 5]).reshape(2, 1)).all()


def test_error_function():
    kvec = np.array([50.0, 50.0, 0.0, 50.0, 50.0])
    depths = [1.0, 2.0]
    pixel_coords = [gtsam.Point2(10, 60), gtsam.Point2(50, 50), gtsam.Point2(30, 40)]
    errors_expected = [gtsam.Point2(5, 5), gtsam.Point2(12, -10), gtsam.Point2(23, 32)]
    predicted_pixel = [
        gtsam.Point2(18.3333, 51.6667),
        gtsam.Point2(38, 60),
        gtsam.Point2(13.6667, 11.3333),
    ]
    kvec = np.array([50.0, 50.0, 0.0, 50.0, 50.0])
    R_w_c = gtsam.Rot3.RzRyRx(-np.pi / 2, 0, -np.pi / 2)
    pose_w_c1 = gtsam.Pose3(R_w_c, gtsam.Point3(2, 1, 0))
    pose_w_c2 = gtsam.Pose3(R_w_c, gtsam.Point3(1.5, 1, 0))
    d_k_1 = gtsam.symbol("d", 1)
    p_k_1 = gtsam.symbol("x", 1)
    p_k_2 = gtsam.symbol("x", 2)

    dba_error = Droid_DBA_Error(kvec)

    for i in range(len(pixel_coords)):
        dba_error.pixel_to_project = pixel_coords[i]
        dba_error.predicted_pixel = predicted_pixel[i]

        error_computed = dba_error.error(
            pose_i=pose_w_c1, pose_j=pose_w_c2, depth_i=depths[0]
        )
        assert error_computed.shape == (2, 1)
        assert_allclose(
            error_computed, errors_expected[i].reshape(2, 1), atol=1e-5, rtol=1e-5
        )

        # check jacobian shapes
        H_pose_1 = np.zeros([2, 6])
        H_pose_2 = np.zeros([2, 6])
        H_d = np.zeros([1, 6])
        error_computed = dba_error.error(
            pose_i=pose_w_c1,
            pose_j=pose_w_c2,
            depth_i=depths[0],
            H_pose_i=H_pose_1,
            H_pose_j=H_pose_2,
            H_depth_i=H_d,
        )
