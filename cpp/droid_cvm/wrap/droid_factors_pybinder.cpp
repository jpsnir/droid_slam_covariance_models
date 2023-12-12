#include <boost/shared_ptr.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl_bind.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>
#include <gtsam/base/serialization.h>
#include <pybind11/iostream.h>
#include <memory>

// Droid factor to bind.
#include "custom_factors/droid_DBA_factor.h"
#include <gtsam/nonlinear/utilities.h> // for RedirectCout.
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/config.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Rot3.h>

namespace py = pybind11;
using namespace droid_factors;
using namespace std;

PYBIND11_DECLARE_HOLDER_TYPE(TYPE_PLACEHOLDER_DONOTUSE, boost::shared_ptr<TYPE_PLACEHOLDER_DONOTUSE>);
DroidDBAFactor construct(
        Key k1, Key k2, Key k3, gtsam::Point2 p1, gtsam::Point2 p2,
        SharedNoiseModel &model){
    // copy of the object.
    //noiseModel::Diagonal::shared_ptr casted_model = model.cast<noiseModel::Diagonal::shared_ptr>();
    return DroidDBAFactor(k1, k2, k3, p1, p2, model);
}


DroidDBAFactor construct1(Key k1, Key k2, Key k3, gtsam::Point2 p1, gtsam::Point2 p2) {
    // copy of the object.
    // different memory after copy operation.
            return DroidDBAFactor(k1, k2, k3, p1, p2);
}

auto construct2(Key k1, Key k2, Key k3, gtsam::Point2 p1, gtsam::Point2 p2) {
    // raii style pointer, with complete ownership.
    // conceptually python should have the same memory
            return std::unique_ptr<DroidDBAFactor>(new DroidDBAFactor(k1, k2, k3, p1, p2));
}

auto construct3(Key k1, Key k2, Key k3, gtsam::Point2 p1, gtsam::Point2 p2) {
    // raw pointer, python takes ownserhip by default unless set.
    // same memory
            return (new DroidDBAFactor(k1, k2, k3, p1, p2));
}

auto construct4(
        Key k1, Key k2, Key k3, gtsam::Point2 p1, gtsam::Point2 p2,
          boost::shared_ptr<Cal3_S2> &K) {
    return boost::shared_ptr<DroidDBAFactor>(new DroidDBAFactor(k1, k2, k3, p1, p2, K));
}

int key(const DroidDBAFactor* f, int id){
    int keyval = -1;
    if (id > 0 && id < f->keys().size())
        keyval = f->keys()[id-1];
    return keyval;
}

int key_o(const DroidDBAFactor &f, int id){
    int keyval = -1;
    if (id > 0 && id < f.keys().size())
        keyval = f.keys()[id-1];
    return keyval;
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
  py::class_<DroidDBAFactor, unique_ptr<DroidDBAFactor>>(m, "DBA")
      .def(py::init<size_t, size_t, size_t, const gtsam::Point2 ,
                    const gtsam::Point2>(),
           py::arg("key1_pi"), py::arg("key2_pj"), py::arg("key3_di"),
           py::arg("pixel_i"), py::arg("predicted_pixel_j"), py::return_value_policy::reference)
      .def("construct0",&construct)
      .def("construct1",&construct1, py::return_value_policy::move)
      .def("construct2",&construct2, py::return_value_policy::move)
      .def("construct3",&construct3, py::return_value_policy::move)
      .def("construct4",&construct4, py::return_value_policy::move)
      .def("key_id", &key)
      .def("key_id1", &key_o)
      .def("compose_rotations",&compose_rotations);


}
//.def(py::init<const Key, const Key, const Key, const gtsam::Point2 ,
//                const gtsam::Point2, const gtsam::Cal3_S2 ,
//                const gtsam::SharedNoiseModel &>(),
//       py::arg("key1_pi"), py::arg("key2_pj"), py::arg("key3_di"),
//       py::arg("pixel_i"), py::arg("predicted_pixel_j"), py::arg("K"),
//       py::arg("noiseModel"));
