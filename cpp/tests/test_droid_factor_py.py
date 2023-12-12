'''
tests for custom factor pybind module
'''

import numpy as np
import gtsam
import sys
import pytest
from pathlib import Path
package_path = Path('/home/jagatpreet/workspaces/NEUFR/vins/covariance_modeling')
sys.path.append(str(package_path.joinpath(
    'droid_slam_covariance_modeling/cpp/build/droid_cvm')))
import droid_factors_py
from droid_factors_py import DBA


def test_pybind_init():
    '''
    checks whether the python package behaves well after building
    with gtsam types.
    '''
    dba_factor = DBA(10, 20, 30, np.array([10., 20.]), np.array([20., 30.]))
    print(f' type of object - py::init : {type(dba_factor)}')
    assert (dba_factor.key_id(1) == 10)
    assert (dba_factor.key_id1(1) == 10)
    assert (dba_factor.key_id(2) == 20)
    assert (dba_factor.key_id1(2) == 20)


@pytest.mark.xfail(raises=TypeError)
@pytest.mark.skip(reason="type conversion of gtsam is not working yet")
def test_object_builder0():
    nm = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.2]))
    dba_factor = DBA.construct0(
        10, 2, 3, np.array([10, 20]), np.array([20, 23]), nm)
    print(f' type of object - constructor 0 {type(dba_factor)}')
    assert (dba_factor.key_id(1) == 10)
    assert (dba_factor.key_id1(1) == 10)


def test_object_builder1():
    dba_factor = DBA.construct1(10, 2, 3, np.array([10, 20]), np.array([20, 23]))
    print(f' type of object - constructor 1 {type(dba_factor)}')
    assert (dba_factor.key_id(1) == 10)
    assert (dba_factor.key_id1(1) == 10)


def test_object_builder2():
    dba_factor = DBA.construct2(12, 22, 32, np.array([10, 20]), np.array([20, 23]))
    print(f' type of object - constructor2 - {type(dba_factor)}')

    assert (dba_factor.key_id(1) == 12)
    assert (dba_factor.key_id1(1) == 12)


def test_object_builder3():
    dba_factor = DBA.construct3(10, 21, 32, np.array([10, 20]), np.array([20, 23]))
    print(f' type of object - constructor 3 {type(dba_factor)}')
    assert (dba_factor.key_id(1) == 10)


@pytest.mark.xfail(raises=TypeError)
@pytest.mark.skip(reason="type conversion of gtsam is not working yet")
def test_object_builder4():
    k = gtsam.Cal3_S2(50., 50., 0, 50., 50.)
    nm = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.2]))
    dba_factor = DBA.construct4(
        10, 21, 32, np.array([10, 20]), np.array([20, 23]), k, nm)
    print(f' type of object - constructor 4 {type(dba_factor)}')
    assert (dba_factor.key_id(1) == 10)


@pytest.mark.skip(reason="type conversion of gtsam is not working yet")
@pytest.mark.xfail(raises=TypeError)
def test_compose_rotations1():
    '''
 testing gtsam custom types other than eigen vectors and scalars.
 '''
    r1 = gtsam.Rot3.RzRyRx(np.pi / 2, np.pi / 3, 0)
    r2 = r1.inverse()
    print(f'compose operation = {r1.compose(r2)}')
    DBA.compose_rotations(r1, r2)


@pytest.mark.skip(reason="type conversion of gtsam is not working yet")
def test_init_with_droid_dba_main_constructor_type():
    '''
    '''
    k = gtsam.Cal3_S2(50., 50., 0, 50., 50.)
    nm = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.2]))
