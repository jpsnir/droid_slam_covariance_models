import pytest
import gtsam
import numpy as np


def test_Cal3_S2_camera_model_constructor():
    cam = gtsam.Cal3_S2()
    print(f"camera matrix = {cam.print()}")
    print(f"type of property: {type(cam.fx)}")
    assert cam.fx() == 1
    assert cam.fy() == 1
    assert cam.skew() == 0
    print(
        f"type of property value principal point:\
            {type(cam.principalPoint())}, shape = {cam.principalPoint().shape}"
    )
    np.testing.assert_equal(cam.principalPoint(), np.array([0.0, 0.0]))
    # initialization with parameters
    cam = gtsam.Cal3_S2(50.0, 50.0, 0, 25.0, 25.0)
    assert cam.fx() == 50.0
    assert cam.fy() == 50.0
    assert cam.skew() == 0
    np.testing.assert_equal(cam.principalPoint(), np.array([25.0, 25.0]))
    print(f"Cam vector function: {cam.vector()} - type - {type(cam.vector())}")
    print(f"cam K function: {cam.K()} - type {type(cam.K())}")


@pytest.mark.xfail
def test_Cal3_S2_invalid_initialization():
    """
    To show that negative entries are allowed in calibration matrix.
    In practise this wont be the case, focal length and principal point
    has to be positive.
    """
    with pytest.raises(Exception) as e_info:
        cam = gtsam.Cal3_S2(-10, -10, 0, 20, 20)
        cam = gtsam.Cal3_S2(0, 0, 0, 20, 20)
        cam = gtsam.Cal3_S2(10, 10, 0, -20, -20)

    with pytest.raises(Exception) as e_info:
        cam = gtsam.Cal3_S2(-50.0, 50.0, 0, -25.0, 25.0)
        ph_cam = gtsam.PinholePoseCal3_S2(
            pose=gtsam.Pose3.Identity(),
            K=cam,
        )


def test_pinhole_camera_model_constructor():
    """
    Perpective camera model
    with projection matrix
    """
    # camera model
    cam = gtsam.Cal3_S2(50.0, 50.0, 0, 25.0, 25.0)
    ph_cam = gtsam.PinholeCameraCal3_S2()
    print(f"pinhole camera with pose: {cam.print()}")
    ph_cam = gtsam.PinholePoseCal3_S2(
        pose=gtsam.Pose3.Identity(),
        K=cam,
    )
    assert (ph_cam.pose().matrix() == gtsam.Pose3.Identity().matrix()).all
    assert isinstance(ph_cam.calibration(), gtsam.Cal3_S2)
    assert (cam.vector() == ph_cam.calibration().vector()).all


def test_Cal3_S2_camera_calibrate_method():
    """
    The output of calibrate method is tested
    The code also illustrates the use of calibrate method,
    what it is actually doing inside.
    """
    cam = gtsam.Cal3_S2(50, 50, 0, 50, 50)
    x = np.linspace(20, 70, 6)
    for sample in x:
        # point on the center line of image plane
        # only x changes.
        pt = gtsam.Point2(sample, cam.py())
        print(
            f"Output of pt({pt}) with  calibrate function:\
                {cam.calibrate(pt)}, shape = {cam.calibrate(pt).shape}"
        )

    p_xy = gtsam.Point2(30, 40)
    normalized_coords = np.array(
        [(p_xy[0] - cam.px()) / cam.fx(), (p_xy[1] - cam.py()) / cam.fy()]
    )
    # We are not checking the jacobian of this method. That can be
    # another test.
    print(f"{cam.calibrate(p_xy)} - {normalized_coords}")
    assert (cam.calibrate(p_xy) == normalized_coords).all()


def test_gtsam_Pose3_functions():
    """
    Understand the behavior of methods from pose3 types.
    world coordinate system (FRD)
    x -> forward, y->left, z -> up
    camera coordinate system
    x -> right, y-> down, z->forward
    """
    # define camera wrt camera0
    R = gtsam.Rot3.RzRyRx(-np.pi / 2, 0, -np.pi / 2)
    assert isinstance(R, gtsam.Rot3)
    T_wc = gtsam.Pose3(r=R, t=np.array([5, 0, 0]))
    l_c = gtsam.Point3(x=0, y=0, z=10)
    print(f"T_wc = {T_wc.print()}")
    # Point in world frame.
    # l_w = T_wc * l_c
    l_w = T_wc.transformFrom(l_c)
    assert isinstance(T_wc, gtsam.Pose3)
    # Landmark is away from camera by 10 units in x axis
    # camera origin is away from world origin by 5 units in xaxis
    assert np.isclose(l_w[0], 15)

    # compose and transformPoseFrom
    delta_T = gtsam.Pose3(r=gtsam.Rot3.Rz(0.1), t=np.array([0, 0, 0.2]))

    T_w_c1 = T_wc.transformPoseFrom(delta_T)
    print(f"transform pose from = {T_w_c1.print()}")
    print(f"Compose = {T_wc.compose(delta_T)}")
    assert T_w_c1.equals(T_wc.compose(delta_T), 0.001)


def test_Cal3_S2_camera_uncalibrate_method():
    """
    The output of uncalibrate method is tested.
    The code illustrates how uncalibrate method converts the
    normalized( or intrinsic [gtsam language] ) to pixel coordinates.
    """

    cam = gtsam.Cal3_S2(50, 50, 0, 50, 50)
    # Optical center is referred as (0,0)
    normalized_coords = np.array(
        [[1, 0], [0, 1], [1, 1], [-1, 0], [0, -1], [-1, -1], [-0.5, -0.5]]
    )
    expected = np.array(
        [[100, 50], [50, 100], [100, 100], [0, 50], [50, 0], [0, 0], [25, 25]]
    )
    # uncalibrate is solving this relation
    # x - cx
    # -------- = u => x = u*fx + cx
    #    fx

    for i, crds in enumerate(normalized_coords):
        pt = gtsam.Point2(crds)
        assert (cam.uncalibrate(pt) == expected[i]).all()


def test_pinhole_perspective_camera_functions():
    """
    test different functions perspective camera.
    project,
    """
    cam = gtsam.Cal3_S2(50, 50, 0, 50, 50)
    # camera's z -> forward, x -> right, y -> down
    # origin of coordinate frame at optical center.
    camera_pose = gtsam.Pose3.Identity()
    ph_camera = gtsam.PinholePoseCal3_S2(
        pose=camera_pose,
        K=cam,
    )

    pt_3d_list = [
        gtsam.Point3(x=0, y=0, z=10),
        gtsam.Point3(x=-5, y=0, z=10),
        gtsam.Point3(x=-5, y=0, z=20),
        gtsam.Point3(x=5, y=0, z=10),
        gtsam.Point3(x=0, y=5, z=10),
        gtsam.Point3(x=0, y=5, z=0.01),
        gtsam.Point3(x=100, y=5, z=5),
    ]
    chiral_pts_3d = [
        gtsam.Point3(x=0, y=5, z=0),
        gtsam.Point3(x=0, y=5, z=-1),
    ]

    for pt_3 in pt_3d_list:
        pt_2d = ph_camera.project(pt_3)
        print(f"camera projection {pt_3} : {pt_2d}")

    with pytest.raises(Exception) as e_info:
        pt_2d = ph_camera.project(chiral_pts_3d[0])
    print(f"{e_info}")
    # projectSafe will check whether the 3D point is behind the camera.
    # useful for avoiding errors
    for pt_3 in chiral_pts_3d:
        pt_2d, flag = ph_camera.projectSafe(pt_3)
        assert flag == False
        print(f"camera projection {pt_3}, flag={flag} : {pt_2d}")
    # Camera projection function is implementing the following
    # x = X/Z * fx + cx
    # y = Y/Z * fy + cy
    # It projects a 3D point to pixel coordinates.
    # NOTE: Pixel coordinate system origin is at corner of image plane
    # cx and cy is measured in image plane.
    # Does not check whether the point is behind camera optical center.
    # Throws Chirality exception in that case
    for i, p in enumerate(pt_3d_list):
        K = ph_camera.calibration()
        expected_xy = np.array(
            [p[0] / p[2] * K.fx() + K.px(), p[1] / p[2] * K.fy() + K.py()]
        )
        assert (expected_xy == ph_camera.project(p)).all()

    # back project a 2D point.
    depth = pt_3d_list[0][2]
    backprj_pt = ph_camera.backproject(ph_camera.project(pt_3d_list[0]), depth)
    assert isinstance(backprj_pt, np.ndarray)
    assert (backprj_pt == pt_3d_list[0]).all()

    # test projection with change in transformation
    T_wc = gtsam.Pose3(
        r=gtsam.Rot3.RzRyRx(-np.pi / 2, 0, -np.pi / 2), t=np.array([5, 0, 0])
    )
    ph_camera_1 = ph_camera
    ph_camera_2 = gtsam.PinholePoseCal3_S2(pose=T_wc, K=cam)
    backprj_pt_c = ph_camera_1.backproject(gtsam.Point2(x=10, y=10), 5)
    backprj_pt_2 = ph_camera_2.backproject(gtsam.Point2(x=10, y=10), 5)
    pt_3d_w = T_wc.transformFrom(backprj_pt_c)
    # This shows that the back projected point from a camera is in the world
    # frame.
    print(f"Point in world frame = {pt_3d_w}")
    print(f"back projected point directly from camera = {backprj_pt_2}")
    assert (pt_3d_w == backprj_pt_2).all()


def test_Cal3_S2_function_jacobians():
    """
    Stub test function to test the jacobian of
    the calibrate and uncalibrate function wrt
    different variables.
    """
    # camera's z -> forward, x -> right, y -> down
    # origin of coordinate frame at optical center.
    T_wc = gtsam.Pose3(
        r=gtsam.Rot3.RzRyRx(-np.pi / 2, 0, -np.pi / 2), t=np.array([5, 0, 0])
    )
    cam = gtsam.Cal3_S2(50, 50, 0, 50, 50)
    ph_camera = gtsam.PinholePoseCal3_S2(pose=T_wc, K=cam)
    # point in world coordinates
    pt_3d_list = [
        gtsam.Point3(x=10, y=0, z=10),
        gtsam.Point3(x=10, y=10, z=0.01),
        gtsam.Point3(x=100, y=5, z=5),
    ]
    Dpose = np.zeros((2, 6), order="F")
    Dpoint = np.zeros((2, 3), order="F")
    Dcal = np.zeros((2, 5), order="F")

    # taken from this test in gtsam
    # https://github.com/borglab/gtsam/blob/develop/python/gtsam/tests/test_PinholeCamera.py
    pt_2d = ph_camera.project(pt_3d_list[1], Dpose, Dpoint, Dcal)
    depth = pt_3d_list[0][1]
    # order is important as
    H_c2 = np.zeros((3, 6), order="F")
    H_p = np.zeros((3, 2), order="F")
    H_d = np.zeros((3, 1), order="F")
    H_cal = np.zeros((3, 5), order="F")
    backprj_pt = ph_camera.backproject(pt_2d, depth)
    assert depth == backprj_pt[2]
    backprj_pt = ph_camera.backproject(pt_2d, depth, H_c2, H_p, H_d, H_cal)
    assert backprj_pt.shape == (3,)
    # so need to reshape
    print(H_c2)
    # print(H_d)
