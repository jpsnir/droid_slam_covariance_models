"""
This module implements the error functions in droid slam paper.
In future, any new functions for computing error can be included in here.
"""

import gtsam
import numpy as np
import typing as T


class Droid_DBA_Error:
    def __init__(self, k_vec: np.ndarray) -> None:
        """
        constructor
        """
        if k_vec.ndim == 1:
            l = k_vec.shape
            assert l == (
                5,
            ), f"Incorrect number of camera parameters-{l} given, 5 required"
        elif k_vec.dim == 2:
            m, n = k_vec.shape
            assert (
                m == 5 and n == 1
            ), "Incorrect number of camera parameters - {m} given , 5 required"
        # create calibration object
        self._K = gtsam.Cal3_S2(k_vec)
        self._predicted_pixel_j = None
        self._pixel_i = None
        self.backprj_pt_w = np.zeros([3, 1])
        self._error = np.zeros([2, 1])

    @property
    def calibration(self):
        return self._K

    @property
    def predicted_pixel(self):
        return self._predicted_pixel_j

    @predicted_pixel.setter
    def predicted_pixel(self, value: np.ndarray):
        if value.shape == (2,):
            value = value.reshape(2, 1)
        assert value.shape == (
            2,
            1,
        ), f"Shape mismatch - required shape (2, 1), given {value.shape}"
        self._predicted_pixel_j = value

    @property
    def pixel_to_project(self):
        return self._pixel_i

    @pixel_to_project.setter
    def pixel_to_project(self, value: np.ndarray):
        if value.shape == (2,):
            value = value.reshape(2, 1)
        assert value.shape == (
            2,
            1,
        ), f"Shape mismatch - required shape (2, 1), given {value.shape}"
        self._pixel_i = value

    @property
    def key_pose_j(self):
        return self._key_pj

    def error(
        self,
        pose_i: gtsam.Pose3,
        pose_j: gtsam.Pose3,
        depth_i: np.float64,
    ) -> np.ndarray:
        """
        Error function with optional Jacobians is implemented
        """
        assert (
            self.pixel_to_project is not None
        ), "pixel in camera i is not set for projection, set it first"
        assert (
            self.predicted_pixel is not None
        ), "predicted pixel in camera j is not set, set it first."
        H_c1 = np.zeros((3, 6), order="F")
        H_pi = np.zeros((3, 2), order="F")
        H_di = np.zeros((3, 1), order="F")
        H_cal1 = np.zeros((3, 5), order="F")
        H_cal2 = np.zeros((2, 5), order="F")
        H_pt_w = np.zeros((3, 3), order="F")
        H_pt_c2 = np.zeros((2, 3), order="F")
        H_c2 = np.zeros((3, 6), order="F")
        H_c2_c2 = np.zeros((2, 6), order="F")

        try:
            camera1 = gtsam.PinholePoseCal3_S2(pose_i, self._K)
            self.backprj_pt_w = camera1.backproject(
                self._pixel_i, depth_i, H_c1, H_pi, H_di, H_cal1
            )
            assert self.backprj_pt_w.shape == (3,)
            self.pt_c2 = pose_j.transformTo(self.backprj_pt_w, H_c2, H_pt_w)
            assert self.pt_c2.shape == (3,)
            pose_c2_c2 = gtsam.Pose3.Identity()
            camera2 = gtsam.PinholeCameraCal3_S2(pose=pose_c2_c2, K=self._K)
            self.reprojected_pt_j = camera2.project(
                self.pt_c2, H_c2_c2, H_pt_c2, H_cal2
            )
            assert self.reprojected_pt_j.shape == (2,)
            assert self._predicted_pixel_j.shape == (2, 1)
            self._error = self.reprojected_pt_j.reshape(2, 1) - self._predicted_pixel_j
            H_pose_i = H_pt_c2 @ H_pt_w @ H_c1
            H_pose_j = H_pt_c2 @ H_c2
            H_depth_i = H_pt_c2 @ H_pt_w @ H_di
            self.H = [H_pose_i, H_pose_j, H_depth_i]
        except RuntimeError as e:
            self._error = np.zeros((2, 1))
            print(e)
        return self._error, self.H
