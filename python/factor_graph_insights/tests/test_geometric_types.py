'''
Tests are exploratory tests to understand different properties
of geometric objects in symforce
'''
import pytest
import symforce
symforce.set_symbolic_api("sympy")
symforce.set_log_level("warning")
import symforce.symbolic as sf
import gtsam
import numpy as np

from symforce.notebook_util import display
from symforce import ops


yaw = np.pi / 4
pitch = 0
roll = 0


def test_Rot3_type():

    rot3_gtsam = gtsam.Rot3.RzRyRx(yaw, pitch, roll)
    assert (isinstance(rot3_gtsam.matrix(), np.ndarray))


@pytest.mark.xfail
def test_sym_Rot3():
    rot3_sym = sf.Rot3.from_yaw_pitch_roll(yaw, pitch, roll)
    assert (isinstance(rot3_sym.to_rotation_matrix(), np.ndarray))
