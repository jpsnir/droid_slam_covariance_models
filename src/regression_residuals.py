'''
This python file create custom factors for regression problems:
The steps are:
    1. Create a residual function using symbolic types from symforce
    2. use

'''

import symforce
symforce.set_symbolic_api("sympy")
symforce.set_log_level("warn")

import symforce.symbolic as sf
from symforce.notebook_util import display
from symforce import codegen
from symforce.codegen import codegen_util
