#include <boost/shared_ptr.hpp>

#include <boost/smart_ptr/shared_ptr.hpp>
#include <gtsam/base/serialization.h>
#include <gtsam/config.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/utilities.h> // for RedirectCout.
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

// Droid factor to bind.
#include "custom_factors/droid_DBA_factor.h"

BOOST_CLASS_EXPORT(gtsam::GenericValue<gtsam::Cal3_S2>)
BOOST_CLASS_EXPORT(gtsam::Cal3_S2)
PYBIND11_DECLARE_HOLDER_TYPE(TYPE_PLACEHOLDER_DONOTUSE,
                             boost::shared_ptr<TYPE_PLACEHOLDER_DONOTUSE>);
namespace py = pybind11;
using namespace droid_factors;
DroidDBAFactor construct(
        Key k1, Key k2, Key k3, gtsam::Point2 p1, gtsam::Point2 p2,
          boost::shared_ptr<gtsam::noiseModel::Base> &model) {
    return DroidDBAFactor(k1, k2, k3, p1, p2, model);
}
DroidDBAFactor construct1(Key k1, Key k2, Key k3, gtsam::Point2 p1, gtsam::Point2 p2) {
            return DroidDBAFactor(k1, k2, k3, p1, p2);
}

auto construct2(Key k1, Key k2, Key k3, gtsam::Point2 p1, gtsam::Point2 p2) {
            return boost::shared_ptr<DroidDBAFactor>(new DroidDBAFactor(k1, k2, k3, p1, p2));
}

void compose_rotations(Rot3 r1, Rot3 r2){
    std::cout << r1.compose(r2) << std::endl;
}

PYBIND11_MODULE(droid_factors_py, m) {
    m.doc() = "pybind wrapper for droid slam dba layer";
    m.def(
        "init", []() { std::cout << "Droid factor constructor\n"; },
        "Constructor of the factor");
    m.def(
        "hello", []() { std::cout << " Hello DROID wrapper\n"; },
        "A function that returns mean of values in eigen vector");
  py::class_<droid_factors::DroidDBAFactor>(m, "DBA")
      .def(py::init<size_t, size_t, size_t, const gtsam::Point2 ,
                    const gtsam::Point2>(),
           py::arg("key1_pi"), py::arg("key2_pj"), py::arg("key3_di"),
           py::arg("pixel_i"), py::arg("predicted_pixel_j"))
      .def("construct",&construct)
      .def("construct1",&construct1)
      .def("construct2",&construct2)
      .def("compose_rotations",&compose_rotations);
}
//.def(py::init<const Key, const Key, const Key, const gtsam::Point2 ,
//                const gtsam::Point2, const gtsam::Cal3_S2 ,
//                const gtsam::SharedNoiseModel &>(),
//       py::arg("key1_pi"), py::arg("key2_pj"), py::arg("key3_di"),
//       py::arg("pixel_i"), py::arg("predicted_pixel_j"), py::arg("K"),
//       py::arg("noiseModel"));
