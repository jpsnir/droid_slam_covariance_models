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


def squared_error_residual(
        x: sf.Scalar,
        y: sf.Scalar,
        a: sf.Scalar) -> sf.V1:
    '''
    squared distance error for a single variable
    linear regression problem
    y = ax

    e = (y - a1*x))^2
    '''

    return sf.V1(sf.Pow(y - a * x, 2))


def squared_2d_error_residual(
        X: sf.V2,
        Y: sf.V2,
        A: sf.M22) -> sf.V2:
    '''
    squared distance error for a two input, two output
    linear regression problem
    Y = Ax
    '''
    return (Y - A * X).multiply_elementwise(Y - A * X)


if __name__ == "__main__":
    x = sf.Symbol("x")
    y = sf.Symbol("y")
    a = sf.Symbol("a")
    f = squared_error_residual(x, y, a)[0]
    fd = f.diff(a)
    fdd = fd.diff(a)
    print(f' f = {f}, fd = {fd}, fdd = {fdd}')
    codegen_obj = codegen.Codegen.function(
        func=squared_error_residual,
        config=codegen.CppConfig()
    )

    metadata = codegen_obj.generate_function(output_dir="gen")

    # with jacobian
    codegen_jac = codegen_obj.with_jacobians(which_args=["a"])
    metadata = codegen_jac.generate_function(output_dir="gen")

    # with jacobian and hessian
    codegen_lin = codegen_obj.with_linearization(which_args=["a"])
    metadata = codegen_lin.generate_function(output_dir="gen")

    codegen_obj1 = codegen.Codegen.function(
        func=squared_2d_error_residual,
        config=codegen.CppConfig()
    )

    metadata = codegen_obj1.generate_function(output_dir="gen1")
    codegen_obj1_jac = codegen_obj1.with_jacobians(which_args=["A"])
    metadata = codegen_obj1_jac.generate_function(output_dir="gen1")
