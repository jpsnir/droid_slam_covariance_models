#include <boost/shared_ptr.hpp>

#include <pybind11/eigen.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pybind11.h>
#include <gtsam/config.h>
#include <gtsam/base/serialization.h>
#include <gtsam/nonlinear/utilities.h>  // for RedirectCout.

#include "custom_factors/droid_DBA_factor.h"
