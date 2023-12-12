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
from droid_factors_py import DBA


dba_factor = DBA(1, 2, 3, np.array([10., 20.]), np.array([20., 30.]))
print(f' type of object - py::init : {type(dba_factor)}')
dba_factor = DBA.construct1(1, 2, 3, np.array([10, 20]), np.array([20, 23]))
print(f' type of object - constructor 1 {type(dba_factor)}')


def test_pybind_functionality():
    '''
        checks whether the python package behaves well after building
        with gtsam types.
        '''
    dba_factor = DBA(1, 2, 3, np.array([10., 20.]), np.array([20., 30.]))
    print(f' type of object - py::init : {type(dba_factor)}')


def test_object_builder1():
    dba_factor = DBA.construct1(1, 2, 3, np.array([10, 20]), np.array([20, 23]))
    print(f' type of object - constructor 1 {type(dba_factor)}')


@pytest.mark.skip(reason='segmentation fault, need to understand the reason')
def test_object_builder2():
    dba_factor = DBA.construct2(1, 2, 3, np.array([10, 20]), np.array([20, 23]))
    print(f' type of object - constructor2 - {type(dba_factor)}')


@pytest.mark.xfail(raises=Exception)
def test_compose_rotations():
    '''
 testing gtsam custom types other than eigen vectors and scalars.
 '''
    r1 = gtsam.Rot3.RzRyRx(np.pi / 2, np.pi / 3, 0)
    r2 = r1.inverse()
    DBA.compose_rotations(r1, r2)


@pytest.mark.skip(reason='functionality for types not working')
def test_init_with_droid_dba_main_constructor_type():
    '''
 '''
    k = gtsam.Cal3_S2(50., 50., 0, 50., 50.)
    nm = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.2]))
