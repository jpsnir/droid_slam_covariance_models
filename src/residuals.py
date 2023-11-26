'''
This python file create custom factors for regression problems:
The steps are:
    1. Create a residual function using symbolic types from symforce
    2. use

'''

import symforce
symforce.set_symbolic_api("sympy")
symforce.set_log_level("warn")
symforce.set_epsilon_to_symbol()

import symforce.symbolic as sf
from symforce.notebook_util import display
from symforce import codegen
from symforce.codegen import codegen_util
from symforce import jacobian_helpers

H = 100
W = 120


def squared_error_residual(
        x: sf.Scalar,
        y: sf.Scalar,
        a: sf.Scalar,
        b: sf.Scalar) -> sf.V1:
    '''
    squared distance error for a single variable
    linear regression problem
    y = ax + b

    e = (y - a*x - b))^2
    '''

    return sf.V1(sf.Pow(y - a * x - b, 2))


def squared_2d_error_residual(
        X: sf.V2,
        Y: sf.V2,
        A: sf.M22,
        B: sf.V2) -> sf.V2:
    '''
    squared distance error for a two input, two output
    linear regression problem
    Y = Ax
    '''
    return (Y - A * X - B).multiply_elementwise(Y - A * X - B)


def optitrack_position_residuals(
        p0_opti: sf.V3,
        p1_opti: sf.V3,
) -> sf.V1:
    '''
    optitrack distance measurements from its frame.
    '''

    return ((p1_opti - p0_opti).norm())


def droid_slam_residual_single(
        dst_img_coords: sf.V2,
        src_img_coords: sf.V2,
        d_src: sf.Scalar,
        w_pose_i: sf.Pose3,
        w_pose_j: sf.Pose3,
        K: sf.LinearCameraCal,
        pixel_confidence: sf.V2,
        epsilon: sf.Scalar,
) -> sf.V1:
    '''
    residual for each pixel in the image.
    Droid slam computes dense depth map of
    reduce size image. The residual here is
    just computing one single term between two images.
    '''
    real_depth = 1 / d_src
    i_pose_j = w_pose_i.inverse() * w_pose_j
    pt3d, status = K.camera_ray_from_pixel(src_img_coords, epsilon)
    if status:
        landmark_i = real_depth * pt3d
        landmark_j = i_pose_j * landmark_i
    reprojected_lm_j, status = K.pixel_from_camera_point(landmark_j, epsilon)
    error = dst_img_coords - reprojected_lm_j
    sigma = sf.Matrix.diag(pixel_confidence)
    mahanalobnis_distance = (
        error.transpose() * sigma.inv() * error
    )
    return mahanalobnis_distance


# We dont have a tensor in symforce
# so it is hard to define the image pair factor directly.
# def droid_slam_residual_image_pair(
#        dst_img_coords: sf.Matrix(H * W, 2),
#        src_img_coords: sf.Matrix(H * W, 2),
#        inverse_depth_src: sf.Matrix(H * W, 1),
#        w_pose_i: sf.Pose3,
#        w_pose_j: sf.Pose3,
#        K: sf.LinearCameraCal,
# ) -> sf.V1:
#    '''
#    for image pair
#    '''
#    real_depth =
#

if __name__ == "__main__":
    folder = '../include/gen'
    x = sf.Symbol("x")
    y = sf.Symbol("y")
    a = sf.Symbol("a")
    b = sf.Symbol("b")
    f = squared_error_residual(x, y, a, b)[0]
    fd = f.diff(a)
    fdd = fd.diff(a)
    print(f' f = {f}, fd = {fd}, fdd = {fdd}')
    codegen_obj = codegen.Codegen.function(
        func=squared_error_residual,
        config=codegen.CppConfig()
    )

    metadata = codegen_obj.generate_function(output_dir=folder)

    # with jacobian
    codegen_jac = codegen_obj.with_jacobians(which_args=["a", "b"])
    metadata = codegen_jac.generate_function(output_dir=folder)

    # with jacobian and hessian
    codegen_lin = codegen_obj.with_linearization(which_args=["a", "b"])
    metadata = codegen_lin.generate_function(output_dir=folder)

    codegen_obj1 = codegen.Codegen.function(
        func=squared_2d_error_residual,
        config=codegen.CppConfig()
    )

    metadata = codegen_obj1.generate_function(
        output_dir=folder
    )
    codegen_obj1_jac = codegen_obj1.with_jacobians(which_args=["A", "B"])
    metadata = codegen_obj1_jac.generate_function(
        output_dir=folder
    )
    codegen_lin = codegen_obj1.with_linearization(which_args=["A", "B"])
    metadata = codegen_lin.generate_function(
        output_dir=folder
    )

    codegen_lin = codegen.Codegen.function(
        func=droid_slam_residual_single,
        config=codegen.CppConfig()
    ).with_linearization(
        which_args=["w_pose_i", "w_pose_j", "d_src"]
    )
    metadata = codegen_lin.generate_function(
        output_dir=folder
    )

    codegen_lin = codegen.Codegen.function(
        func=droid_slam_residual_single,
        config=codegen.PythonConfig()
    ).with_linearization(
        which_args=["w_pose_i", "w_pose_j", "d_src"]
    )
    metadata = codegen_lin.generate_function(
        output_dir="."
    )
