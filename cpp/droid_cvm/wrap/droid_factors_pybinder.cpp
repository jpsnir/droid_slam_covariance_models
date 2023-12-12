#include <boost/shared_ptr.hpp>

#include <gtsam/base/serialization.h>
#include <gtsam/config.h>
#include <gtsam/nonlinear/utilities.h> // for RedirectCout.
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include "custom_factors/droid_DBA_factor.h"

namespace py = pybind11;
PYBIND11_MODULE(droid_dba_factors_py, m) {
  m.doc() = "pybind wrapper for droid slam dba layer";
    m.def("init",[](){std::cout << "Droid factor constructor\n";},
            "Constructor of the factor");
    m.def(
        "hello", []() { std::cout << " Hello DROID wrapper\n"; },
        "A function that returns mean of values in eigen vector");
}
