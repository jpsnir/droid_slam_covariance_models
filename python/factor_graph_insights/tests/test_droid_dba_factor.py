import pytest
from factor_graph_insights.custom_factors.droid_error_functions import Droid_DBA_Error
import numpy as np


def test_constructor():
    kvec = np.array([50., 50., 0., 50., 50.])
    dba_error = Droid_DBA_Error(kvec)
    dba_error.predicted_pixel = np.array([10, 10]).reshape(2, 1)
    dba_error.pixel_to_project = np.array([4, 5]).reshape(2, 1)

    assert (
        (dba_error.predicted_pixel == np.array([10, 10]).reshape(2, 1)).all()
    )
    assert (
        (dba_error.calibration.vector() == kvec).all()
    )
    assert (
        (dba_error.pixel_to_project == np.array([4, 5]).reshape(2, 1)).all()
    )


def test_error_function()


kvec = np.array([50., 50., 0., 50., 50.])
pose_i
