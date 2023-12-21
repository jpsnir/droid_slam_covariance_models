""" Replicate tests written down in c++ to check whether the 
function implemented has the same validity.
"""
import pytest
from factor_graph_insights.custom_factors.droid_error_functions import Droid_DBA_Error
import numpy as np
from numpy.testing import assert_allclose
import gtsam
from numerical_derivative_py import numerical_derivative_dba as num_dba


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

        error_computed, H = dba_error.error(
            pose_i=pose_w_c1, pose_j=pose_w_c2, depth_i=depths[0]
        )
        assert error_computed.shape == (2, 1)
        assert_allclose(
            error_computed, errors_expected[i].reshape(2, 1), atol=1e-5, rtol=1e-5
        )

        # check jacobian shapes
        error_computed, H = dba_error.error(
            pose_i=pose_w_c1,
            pose_j=pose_w_c2,
            depth_i=depths[0],
        )
        assert H[0].shape == (2, 6)
        assert H[1].shape == (2, 6)
        assert H[2].shape == (2, 1)


def test_droid_dba_custom_factor():
    kvec = np.array([50.0, 50.0, 0.0, 50.0, 50.0])
    depth = 1.0
    pixel_coords = gtsam.Point2(10, 60)
    pixel_confidence = np.array([0.9, 0.9])
    errors_expected = gtsam.Point2(5, 5)
    predicted_pixel = gtsam.Point2(18.3333, 51.6667)
    R_w_c = gtsam.Rot3.RzRyRx(-np.pi / 2, 0, -np.pi / 2)
    pose_w_c1 = gtsam.Pose3(R_w_c, gtsam.Point3(2, 1, 0))
    pose_w_c2 = gtsam.Pose3(R_w_c, gtsam.Point3(1.5, 1, 0))
    d_k_1 = gtsam.symbol("d", 1)
    p_k_1 = gtsam.symbol("x", 1)
    p_k_2 = gtsam.symbol("x", 2)
    droid_dba_error = Droid_DBA_Error(kvec)
    droid_dba_error.pixel_to_project = pixel_coords
    droid_dba_error.predicted_pixel = predicted_pixel
    droid_dba_error.make_custom_factor(
        (p_k_1, p_k_2, d_k_1),
        pixel_confidence,
    )

    assert isinstance(droid_dba_error.custom_factor, gtsam.CustomFactor)
    assert_allclose(droid_dba_error.pixel_to_project, pixel_coords.reshape(2, 1))
    assert_allclose(droid_dba_error.predicted_pixel, predicted_pixel.reshape(2, 1))

    # check keys
    assert droid_dba_error.symbols == (p_k_1, p_k_2, d_k_1)
    assert tuple(droid_dba_error.custom_factor.keys()) == (p_k_1, p_k_2, d_k_1)

    # check error
    error_expected, Jacobian = droid_dba_error.error(pose_w_c1, pose_w_c2, depth)
    # mahanalobnis distance = 0.5 * e' * I * e, I is information matrix
    mh_dist = 0.5 * error_expected.T @ np.diag(pixel_confidence) @ error_expected
    values = gtsam.Values()
    values.insert(p_k_1, pose_w_c1)
    values.insert(p_k_2, pose_w_c2)
    values.insert(d_k_1, depth)
    assert_allclose(
        droid_dba_error.custom_factor.unwhitenedError(values).reshape(2, 1),
        error_expected,
    )
    # error is the mahanalobnis distance
    assert_allclose(
        droid_dba_error.custom_factor.error(values),
        mh_dist,
    )


@pytest.mark.skip(
    reason="The pybind module crashes when derivative is extracted, figure out eigen matrices"
)
def test_derivative():
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
    dba_error = Droid_DBA_Error(kvec)
    dba_error.pixel_to_project = pixel_coords[0]
    dba_error.predicted_pixel = predicted_pixel[0]
    R_w_c = gtsam.Rot3.RzRyRx(-np.pi / 2, 0, -np.pi / 2)
    pose_w_c1 = gtsam.Pose3(R_w_c, gtsam.Point3(2, 1, 0))
    pose_w_c2 = gtsam.Pose3(R_w_c, gtsam.Point3(1.5, 1, 0))
    error_computed, H = dba_error.error(
        pose_i=pose_w_c1,
        pose_j=pose_w_c2,
        depth_i=depths[0],
    )
    print(f" Derivate pose 1: {H[0]}")
    print(f" Derivate pose 2: {H[1]}")
    print(f" Derivate depth: {H[2]}")
    cov_mat = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.2])).R()
    numH_pose_1 = np.zeros([2, 6], order="F")
    numH_pose_2 = np.zeros([2, 6], order="F")
    numH_d = np.zeros([1, 6], order="F")
    num_dba(
        pose_w_c1.matrix(),
        pose_w_c2.matrix(),
        depths[0],
        pixel_coords[0],
        predicted_pixel[0],
        cov_mat,
        kvec,
        numH_pose_1,
        numH_pose_2,
        numH_d,
    )
    assert_allclose(numH_pose_1, H[0], rtol=1e-5, atol=1e-5)
    assert_allclose(numH_pose_2, H[1], rtol=1e-5, atol=1e-5)
    assert_allclose(numH_d, H[2], rtol=1e-5, atol=1e-5)
