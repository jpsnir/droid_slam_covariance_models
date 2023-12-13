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
typedef gtsam::NoiseModelFactor3<Pose3, Pose3, double> BaseTernaryFactor;


PYBIND11_DECLARE_HOLDER_TYPE(TYPE_PLACEHOLDER_DONOTUSE, boost::shared_ptr<TYPE_PLACEHOLDER_DONOTUSE>);

DroidDBAFactor construct(
        Key k1, Key k2, Key k3, gtsam::Point2 p1, gtsam::Point2 p2,
        const boost::shared_ptr<gtsam::noiseModel::Base> &model){
    // copy of the object.
    //noiseModel::Diagonal::shared_ptr casted_model = model.cast<noiseModel::Diagonal::shared_ptr>();
    return DroidDBAFactor(k1, k2, k3, p1, p2);
}


DroidDBAFactor construct1(Key k1, Key k2, Key k3, gtsam::Point2 p1, gtsam::Point2 p2) {
    // copy of the object.
    // different memory after copy operation.
            return DroidDBAFactor(k1, k2, k3, p1, p2);
}

auto construct2(Key k1, Key k2, Key k3, gtsam::Point2 p1, gtsam::Point2 p2) {
    // raii style pointer, with complete ownership.
    // conceptually python should have the same memory
            return std::shared_ptr<DroidDBAFactor>(new DroidDBAFactor(k1, k2, k3, p1, p2));
}

auto construct3(Key k1, Key k2, Key k3, gtsam::Point2 p1, gtsam::Point2 p2) {
    // raw pointer, python takes ownserhip by default unless set.
    // same memory
            return (new DroidDBAFactor(k1, k2, k3, p1, p2));
}

      //.def(py::init<size_t, size_t, size_t,
      //              const gtsam::Point2, const gtsam::Point2,
      //              const gtsam::Vector5 >(),
      //     py::arg("key1_pi"), py::arg("key2_pj"), py::arg("key3_di"),
      //     py::arg("pixel_i"), py::arg("predicted_pixel_j"), py::arg("K_vec"))
//auto construct4(
//        Key k1, Key k2, Key k3, gtsam::Point2 p1, gtsam::Point2 p2,
//          Vector5 &K) {
//    return boost::shared_ptr<DroidDBAFactor>(new DroidDBAFactor(k1, k2, k3, p1, p2, K));
//}
auto construct4(
        Key k1, Key k2, Key k3, gtsam::Point2 p1, gtsam::Point2 p2,
          boost::shared_ptr<Cal3_S2> &K) {
    return boost::shared_ptr<DroidDBAFactor>(new DroidDBAFactor(k1, k2, k3, p1, p2, K));
}
auto construct5(Key k1, Key k2, Key k3, gtsam::Point2 p1, gtsam::Point2 p2) {
    // raii style pointer, with complete ownership.
    // conceptually python shBaould have the same memory
            return boost::shared_ptr<DroidDBAFactor>(new DroidDBAFactor(k1, k2, k3, p1, p2));
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

auto calibration(const DroidDBAFactor *f){
    return f->calibration();
}

void compose_rotations(Rot3 r1, Rot3 r2){
    std::cout << r1.compose(r2) << std::endl;
}

PYBIND11_MODULE(droid_factors_py, m) {
    m.doc() = "pybind wrapper for droid slam dba layer";
  py::class_<DroidDBAFactor, boost::shared_ptr<DroidDBAFactor>>(m, "DBA")
      .def(py::init<const Key&, const Key&, const Key&, const gtsam::Point2& ,
                        const gtsam::Point2&, const boost::shared_ptr<gtsam::Cal3_S2>& ,
                        const gtsam::SharedNoiseModel &>(),
               py::arg("key1_pi"), py::arg("key2_pj"), py::arg("key3_di"),
               py::arg("pixel_i"), py::arg("predicted_pixel_j"), py::arg("K"),
               py::arg("noiseModel"))
      .def(
           py::init<size_t, size_t, size_t, const gtsam::Point2 ,
                    const gtsam::Point2>(),
           py::arg("key1_pi"), py::arg("key2_pj"), py::arg("key3_di"),
           py::arg("pixel_i"), py::arg("predicted_pixel_j"), py::return_value_policy::reference)
      .def(py::init<size_t, size_t, size_t,
                    const gtsam::Point2, const gtsam::Point2,
                    const boost::shared_ptr<gtsam::Cal3_S2> >(),
           py::arg("key1_pi"), py::arg("key2_pj"), py::arg("key3_di"),
           py::arg("pixel_i"), py::arg("predicted_pixel_j"), py::arg("K"))
      .def("construct0",&construct)
      .def("construct1",&construct1, py::return_value_policy::move)
      .def("construct2",&construct2)
      .def("construct3",&construct3, py::return_value_policy::move)
      .def("construct4",&construct4, py::return_value_policy::move)
      .def("construct5",&construct5)
      .def("key_id", &key)
      .def("key_id1", &key_o)
      .def("calibration", &calibration)
      .def("compose_rotations",&compose_rotations);

    py::class_<gtsam::Quaternion, boost::shared_ptr<gtsam::Quaternion>>(m, "Quaternion")
        .def("w",[](gtsam::Quaternion* self){return self->w();})
        .def("x",[](gtsam::Quaternion* self){return self->x();})
        .def("y",[](gtsam::Quaternion* self){return self->y();})
        .def("z",[](gtsam::Quaternion* self){return self->z();})
        .def("coeffs",[](gtsam::Quaternion* self){return self->coeffs();});

    py::class_<gtsam::Rot3, boost::shared_ptr<gtsam::Rot3>>(m, "Rot3")
        .def(py::init<>())
        .def(py::init<const gtsam::Matrix&>(), py::arg("R"))
        .def(py::init<const gtsam::Point3&, const gtsam::Point3&, const gtsam::Point3&>(), py::arg("col1"), py::arg("col2"), py::arg("col3"))
        .def(py::init<double, double, double, double, double, double, double, double, double>(), py::arg("R11"), py::arg("R12"), py::arg("R13"), py::arg("R21"), py::arg("R22"), py::arg("R23"), py::arg("R31"), py::arg("R32"), py::arg("R33"))
        .def(py::init<double, double, double, double>(), py::arg("w"), py::arg("x"), py::arg("y"), py::arg("z"))
        .def("print",[](gtsam::Rot3* self, string s){ py::scoped_ostream_redirect output; self->print(s);}, py::arg("s") = "")
        .def("__repr__",
                    [](const gtsam::Rot3& self, string s){
                        gtsam::RedirectCout redirect;
                        self.print(s);
                        return redirect.str();
                    }, py::arg("s") = "")
        .def("equals",[](gtsam::Rot3* self, const gtsam::Rot3& rot, double tol){return self->equals(rot, tol);}, py::arg("rot"), py::arg("tol"))
        .def("inverse",[](gtsam::Rot3* self){return self->inverse();})
        .def("compose",[](gtsam::Rot3* self, const gtsam::Rot3& p2){return self->compose(p2);}, py::arg("p2"))
        .def("between",[](gtsam::Rot3* self, const gtsam::Rot3& p2){return self->between(p2);}, py::arg("p2"))
        .def("retract",[](gtsam::Rot3* self, const gtsam::Vector& v){return self->retract(v);}, py::arg("v"))
        .def("localCoordinates",[](gtsam::Rot3* self, const gtsam::Rot3& p){return self->localCoordinates(p);}, py::arg("p"))
        .def("rotate",[](gtsam::Rot3* self, const gtsam::Point3& p){return self->rotate(p);}, py::arg("p"))
        .def("unrotate",[](gtsam::Rot3* self, const gtsam::Point3& p){return self->unrotate(p);}, py::arg("p"))
        .def("logmap",[](gtsam::Rot3* self, const gtsam::Rot3& p){return self->logmap(p);}, py::arg("p"))
        .def("matrix",[](gtsam::Rot3* self){return self->matrix();})
        .def("transpose",[](gtsam::Rot3* self){return self->transpose();})
        .def("column",[](gtsam::Rot3* self, size_t index){return self->column(index);}, py::arg("index"))
        .def("xyz",[](gtsam::Rot3* self){return self->xyz();})
        .def("ypr",[](gtsam::Rot3* self){return self->ypr();})
        .def("rpy",[](gtsam::Rot3* self){return self->rpy();})
        .def("roll",[](gtsam::Rot3* self){return self->roll();})
        .def("pitch",[](gtsam::Rot3* self){return self->pitch();})
        .def("yaw",[](gtsam::Rot3* self){return self->yaw();})
        .def("axisAngle",[](gtsam::Rot3* self){return self->axisAngle();})
        .def("toQuaternion",[](gtsam::Rot3* self){return self->toQuaternion();})
        .def("quaternion",[](gtsam::Rot3* self){return self->quaternion();})
        .def("slerp",[](gtsam::Rot3* self, double t, const gtsam::Rot3& other){return self->slerp(t, other);}, py::arg("t"), py::arg("other"))
        .def("serialize", [](gtsam::Rot3* self){ return gtsam::serialize(*self); })
        .def("deserialize", [](gtsam::Rot3* self, string serialized){ gtsam::deserialize(serialized, *self); }, py::arg("serialized"))
        .def(py::pickle(
            [](const gtsam::Rot3 &a){ /* __getstate__: Returns a string that encodes the state of the object */ return py::make_tuple(gtsam::serialize(a)); },
            [](py::tuple t){ /* __setstate__ */ gtsam::Rot3 obj; gtsam::deserialize(t[0].cast<std::string>(), obj); return obj; }))
        .def_static("Rx",[](double t){return gtsam::Rot3::Rx(t);}, py::arg("t"))
        .def_static("Ry",[](double t){return gtsam::Rot3::Ry(t);}, py::arg("t"))
        .def_static("Rz",[](double t){return gtsam::Rot3::Rz(t);}, py::arg("t"))
        .def_static("RzRyRx",[](double x, double y, double z){return gtsam::Rot3::RzRyRx(x, y, z);}, py::arg("x"), py::arg("y"), py::arg("z"))
        .def_static("RzRyRx",[](const gtsam::Vector& xyz){return gtsam::Rot3::RzRyRx(xyz);}, py::arg("xyz"))
        .def_static("Yaw",[](double t){return gtsam::Rot3::Yaw(t);}, py::arg("t"))
        .def_static("Pitch",[](double t){return gtsam::Rot3::Pitch(t);}, py::arg("t"))
        .def_static("Roll",[](double t){return gtsam::Rot3::Roll(t);}, py::arg("t"))
        .def_static("Ypr",[](double y, double p, double r){return gtsam::Rot3::Ypr(y, p, r);}, py::arg("y"), py::arg("p"), py::arg("r"))
        .def_static("Quaternion",[](double w, double x, double y, double z){return gtsam::Rot3::Quaternion(w, x, y, z);}, py::arg("w"), py::arg("x"), py::arg("y"), py::arg("z"))
        .def_static("AxisAngle",[](const gtsam::Point3& axis, double angle){return gtsam::Rot3::AxisAngle(axis, angle);}, py::arg("axis"), py::arg("angle"))
        .def_static("Rodrigues",[](const gtsam::Vector& v){return gtsam::Rot3::Rodrigues(v);}, py::arg("v"))
        .def_static("Rodrigues",[](double wx, double wy, double wz){return gtsam::Rot3::Rodrigues(wx, wy, wz);}, py::arg("wx"), py::arg("wy"), py::arg("wz"))
        .def_static("ClosestTo",[](const gtsam::Matrix& M){return gtsam::Rot3::ClosestTo(M);}, py::arg("M"))
        .def_static("identity",[](){return gtsam::Rot3::identity();})
        .def_static("Expmap",[](const gtsam::Vector& v){return gtsam::Rot3::Expmap(v);}, py::arg("v"))
        .def_static("Logmap",[](const gtsam::Rot3& p){return gtsam::Rot3::Logmap(p);}, py::arg("p"))
        .def(py::self * py::self);

    py::class_<gtsam::Pose3, boost::shared_ptr<gtsam::Pose3>>(m, "Pose3")
        .def(py::init<>())
        .def(py::init<const gtsam::Pose3&>(), py::arg("other"))
        .def(py::init<const gtsam::Rot3&, const gtsam::Point3&>(), py::arg("r"), py::arg("t"))
        .def(py::init<const gtsam::Pose2&>(), py::arg("pose2"))
        .def(py::init<const gtsam::Matrix&>(), py::arg("mat"))
        .def("print",[](gtsam::Pose3* self, string s){ py::scoped_ostream_redirect output; self->print(s);}, py::arg("s") = "")
        .def("__repr__",
                    [](const gtsam::Pose3& self, string s){
                        gtsam::RedirectCout redirect;
                        self.print(s);
                        return redirect.str();
                    }, py::arg("s") = "")
        .def("equals",[](gtsam::Pose3* self, const gtsam::Pose3& pose, double tol){return self->equals(pose, tol);}, py::arg("pose"), py::arg("tol"))
        .def("inverse",[](gtsam::Pose3* self){return self->inverse();})
        .def("compose",[](gtsam::Pose3* self, const gtsam::Pose3& pose){return self->compose(pose);}, py::arg("pose"))
        .def("between",[](gtsam::Pose3* self, const gtsam::Pose3& pose){return self->between(pose);}, py::arg("pose"))
        .def("retract",[](gtsam::Pose3* self, const gtsam::Vector& v){return self->retract(v);}, py::arg("v"))
        .def("localCoordinates",[](gtsam::Pose3* self, const gtsam::Pose3& pose){return self->localCoordinates(pose);}, py::arg("pose"))
        .def("expmap",[](gtsam::Pose3* self, const gtsam::Vector& v){return self->expmap(v);}, py::arg("v"))
        .def("logmap",[](gtsam::Pose3* self, const gtsam::Pose3& pose){return self->logmap(pose);}, py::arg("pose"))
        .def("AdjointMap",[](gtsam::Pose3* self){return self->AdjointMap();})
        .def("Adjoint",[](gtsam::Pose3* self, const gtsam::Vector& xi){return self->Adjoint(xi);}, py::arg("xi"))
        .def("transformFrom",[](gtsam::Pose3* self, const gtsam::Point3& point){return self->transformFrom(point);}, py::arg("point"))
        .def("transformTo",[](gtsam::Pose3* self, const gtsam::Point3& point){return self->transformTo(point);}, py::arg("point"))
        .def("rotation",[](gtsam::Pose3* self){return self->rotation();})
        .def("translation",[](gtsam::Pose3* self){return self->translation();})
        .def("x",[](gtsam::Pose3* self){return self->x();})
        .def("y",[](gtsam::Pose3* self){return self->y();})
        .def("z",[](gtsam::Pose3* self){return self->z();})
        .def("matrix",[](gtsam::Pose3* self){return self->matrix();})
        .def("transformPoseFrom",[](gtsam::Pose3* self, const gtsam::Pose3& pose){return self->transformPoseFrom(pose);}, py::arg("pose"))
        .def("transformPoseTo",[](gtsam::Pose3* self, const gtsam::Pose3& pose){return self->transformPoseTo(pose);}, py::arg("pose"))
        .def("range",[](gtsam::Pose3* self, const gtsam::Point3& point){return self->range(point);}, py::arg("point"))
        .def("range",[](gtsam::Pose3* self, const gtsam::Pose3& pose){return self->range(pose);}, py::arg("pose"))
        .def("serialize", [](gtsam::Pose3* self){ return gtsam::serialize(*self); })
        .def("deserialize", [](gtsam::Pose3* self, string serialized){ gtsam::deserialize(serialized, *self); }, py::arg("serialized"))
        .def(py::pickle(
            [](const gtsam::Pose3 &a){ /* __getstate__: Returns a string that encodes the state of the object */ return py::make_tuple(gtsam::serialize(a)); },
            [](py::tuple t){ /* __setstate__ */ gtsam::Pose3 obj; gtsam::deserialize(t[0].cast<std::string>(), obj); return obj; }))
        .def_static("identity",[](){return gtsam::Pose3::identity();})
        .def_static("Expmap",[](const gtsam::Vector& v){return gtsam::Pose3::Expmap(v);}, py::arg("v"))
        .def_static("Logmap",[](const gtsam::Pose3& pose){return gtsam::Pose3::Logmap(pose);}, py::arg("pose"))
        .def_static("adjointMap",[](const gtsam::Vector& xi){return gtsam::Pose3::adjointMap(xi);}, py::arg("xi"))
        .def_static("adjoint",[](const gtsam::Vector& xi, const gtsam::Vector& y){return gtsam::Pose3::adjoint(xi, y);}, py::arg("xi"), py::arg("y"))
        .def_static("adjointMap_",[](const gtsam::Vector& xi){return gtsam::Pose3::adjointMap_(xi);}, py::arg("xi"))
        .def_static("adjoint_",[](const gtsam::Vector& xi, const gtsam::Vector& y){return gtsam::Pose3::adjoint_(xi, y);}, py::arg("xi"), py::arg("y"))
        .def_static("adjointTranspose",[](const gtsam::Vector& xi, const gtsam::Vector& y){return gtsam::Pose3::adjointTranspose(xi, y);}, py::arg("xi"), py::arg("y"))
        .def_static("ExpmapDerivative",[](const gtsam::Vector& xi){return gtsam::Pose3::ExpmapDerivative(xi);}, py::arg("xi"))
        .def_static("LogmapDerivative",[](const gtsam::Pose3& xi){return gtsam::Pose3::LogmapDerivative(xi);}, py::arg("xi"))
        .def_static("wedge",[](double wx, double wy, double wz, double vx, double vy, double vz){return gtsam::Pose3::wedge(wx, wy, wz, vx, vy, vz);}, py::arg("wx"), py::arg("wy"), py::arg("wz"), py::arg("vx"), py::arg("vy"), py::arg("vz"))
        .def(py::self * py::self);

        py::class_<gtsam::Cal3_S2, boost::shared_ptr<gtsam::Cal3_S2>>(m, "Cal3_S2")
        .def(py::init<>())
        .def(py::init<double, double, double, double, double>(), py::arg("fx"), py::arg("fy"), py::arg("s"), py::arg("u0"), py::arg("v0"))
        .def(py::init<const gtsam::Vector&>(), py::arg("v"))
        .def(py::init<double, int, int>(), py::arg("fov"), py::arg("w"), py::arg("h"))
        .def("print",[](gtsam::Cal3_S2* self, string s){ py::scoped_ostream_redirect output; self->print(s);}, py::arg("s") = "Cal3_S2")
        .def("__repr__",
                    [](const gtsam::Cal3_S2& self, string s){
                        gtsam::RedirectCout redirect;
                        self.print(s);
                        return redirect.str();
                    }, py::arg("s") = "Cal3_S2")
        .def("equals",[](gtsam::Cal3_S2* self, const gtsam::Cal3_S2& rhs, double tol){return self->equals(rhs, tol);}, py::arg("rhs"), py::arg("tol"))
        .def("dim",[](gtsam::Cal3_S2* self){return self->dim();})
        .def("retract",[](gtsam::Cal3_S2* self, const gtsam::Vector& v){return self->retract(v);}, py::arg("v"))
        .def("localCoordinates",[](gtsam::Cal3_S2* self, const gtsam::Cal3_S2& c){return self->localCoordinates(c);}, py::arg("c"))
        .def("calibrate",[](gtsam::Cal3_S2* self, const gtsam::Point2& p){return self->calibrate(p);}, py::arg("p"))
        .def("uncalibrate",[](gtsam::Cal3_S2* self, const gtsam::Point2& p){return self->uncalibrate(p);}, py::arg("p"))
        .def("fx",[](gtsam::Cal3_S2* self){return self->fx();})
        .def("fy",[](gtsam::Cal3_S2* self){return self->fy();})
        .def("skew",[](gtsam::Cal3_S2* self){return self->skew();})
        .def("px",[](gtsam::Cal3_S2* self){return self->px();})
        .def("py",[](gtsam::Cal3_S2* self){return self->py();})
        .def("principalPoint",[](gtsam::Cal3_S2* self){return self->principalPoint();})
        .def("vector",[](gtsam::Cal3_S2* self){return self->vector();})
        .def("K",[](gtsam::Cal3_S2* self){return self->K();})
        .def("serialize", [](gtsam::Cal3_S2* self){ return gtsam::serialize(*self); })
        .def("deserialize", [](gtsam::Cal3_S2* self, string serialized){ gtsam::deserialize(serialized, *self); }, py::arg("serialized"))
        .def(py::pickle(
            [](const gtsam::Cal3_S2 &a){ /* __getstate__: Returns a string that encodes the state of the object */ return py::make_tuple(gtsam::serialize(a)); },
            [](py::tuple t){ /* __setstate__ */ gtsam::Cal3_S2 obj; gtsam::deserialize(t[0].cast<std::string>(), obj); return obj; }))
        .def_static("Dim",[](){return gtsam::Cal3_S2::Dim();});

    ///////////////////////////////////////////////////////////////////

    pybind11::module m_noiseModel = m.def_submodule("noiseModel", "noiseModel submodule");

    py::class_<gtsam::noiseModel::Base, boost::shared_ptr<gtsam::noiseModel::Base>>(m_noiseModel, "Base")
        .def("print_",[](gtsam::noiseModel::Base* self, string s){ self->print(s);}, py::arg("s"))
        .def("__repr__",
                    [](const gtsam::noiseModel::Base &a) {
                        gtsam::RedirectCout redirect;
                        a.print("");
                        return redirect.str();
                    });

    py::class_<gtsam::noiseModel::Gaussian, gtsam::noiseModel::Base, boost::shared_ptr<gtsam::noiseModel::Gaussian>>(m_noiseModel, "Gaussian")
        .def("equals",[](gtsam::noiseModel::Gaussian* self, gtsam::noiseModel::Base& expected, double tol){return self->equals(expected, tol);}, py::arg("expected"), py::arg("tol"))
        .def("R",[](gtsam::noiseModel::Gaussian* self){return self->R();})
        .def("information",[](gtsam::noiseModel::Gaussian* self){return self->information();})
        .def("covariance",[](gtsam::noiseModel::Gaussian* self){return self->covariance();})
        .def("whiten",[](gtsam::noiseModel::Gaussian* self,const gtsam::Vector& v){return self->whiten(v);}, py::arg("v"))
        .def("unwhiten",[](gtsam::noiseModel::Gaussian* self,const gtsam::Vector& v){return self->unwhiten(v);}, py::arg("v"))
        .def("Whiten",[](gtsam::noiseModel::Gaussian* self,const gtsam::Matrix& H){return self->Whiten(H);}, py::arg("H"))
.def("serialize",
    [](gtsam::noiseModel::Gaussian* self){
        return gtsam::serialize(self);
    }
)
.def("deserialize",
    [](gtsam::noiseModel::Gaussian* self, string serialized){
        return gtsam::deserialize(serialized, self);
    })

        .def_static("Information",[](const gtsam::Matrix& R){return gtsam::noiseModel::Gaussian::Information(R);}, py::arg("R"))
        .def_static("SqrtInformation",[](const gtsam::Matrix& R){return gtsam::noiseModel::Gaussian::SqrtInformation(R);}, py::arg("R"))
        .def_static("Covariance",[](const gtsam::Matrix& R){return gtsam::noiseModel::Gaussian::Covariance(R);}, py::arg("R"));

    py::class_<gtsam::noiseModel::Diagonal, gtsam::noiseModel::Gaussian, boost::shared_ptr<gtsam::noiseModel::Diagonal>>(m_noiseModel, "Diagonal")
        .def("R",[](gtsam::noiseModel::Diagonal* self){return self->R();})
        .def("sigmas",[](gtsam::noiseModel::Diagonal* self){return self->sigmas();})
        .def("invsigmas",[](gtsam::noiseModel::Diagonal* self){return self->invsigmas();})
        .def("precisions",[](gtsam::noiseModel::Diagonal* self){return self->precisions();})
.def("serialize",
    [](gtsam::noiseModel::Diagonal* self){
        return gtsam::serialize(self);
    }
)
.def("deserialize",
    [](gtsam::noiseModel::Diagonal* self, string serialized){
        return gtsam::deserialize(serialized, self);
    })

        .def_static("Sigmas",[](const gtsam::Vector& sigmas){return gtsam::noiseModel::Diagonal::Sigmas(sigmas);}, py::arg("sigmas"))
        .def_static("Variances",[](const gtsam::Vector& variances){return gtsam::noiseModel::Diagonal::Variances(variances);}, py::arg("variances"))
        .def_static("Precisions",[](const gtsam::Vector& precisions){return gtsam::noiseModel::Diagonal::Precisions(precisions);}, py::arg("precisions"));

    py::class_<gtsam::noiseModel::Constrained, gtsam::noiseModel::Diagonal, boost::shared_ptr<gtsam::noiseModel::Constrained>>(m_noiseModel, "Constrained")
        .def("unit",[](gtsam::noiseModel::Constrained* self){return self->unit();})
.def("serialize",
    [](gtsam::noiseModel::Constrained* self){
        return gtsam::serialize(self);
    }
)
.def("deserialize",
    [](gtsam::noiseModel::Constrained* self, string serialized){
        return gtsam::deserialize(serialized, self);
    })

        .def_static("MixedSigmas",[](const gtsam::Vector& mu,const gtsam::Vector& sigmas){return gtsam::noiseModel::Constrained::MixedSigmas(mu, sigmas);}, py::arg("mu"), py::arg("sigmas"))
        .def_static("MixedSigmas",[]( double m,const gtsam::Vector& sigmas){return gtsam::noiseModel::Constrained::MixedSigmas(m, sigmas);}, py::arg("m"), py::arg("sigmas"))
        .def_static("MixedVariances",[](const gtsam::Vector& mu,const gtsam::Vector& variances){return gtsam::noiseModel::Constrained::MixedVariances(mu, variances);}, py::arg("mu"), py::arg("variances"))
        .def_static("MixedVariances",[](const gtsam::Vector& variances){return gtsam::noiseModel::Constrained::MixedVariances(variances);}, py::arg("variances"))
        .def_static("MixedPrecisions",[](const gtsam::Vector& mu,const gtsam::Vector& precisions){return gtsam::noiseModel::Constrained::MixedPrecisions(mu, precisions);}, py::arg("mu"), py::arg("precisions"))
        .def_static("MixedPrecisions",[](const gtsam::Vector& precisions){return gtsam::noiseModel::Constrained::MixedPrecisions(precisions);}, py::arg("precisions"))
        .def_static("All",[]( size_t dim){return gtsam::noiseModel::Constrained::All(dim);}, py::arg("dim"))
        .def_static("All",[]( size_t dim, double mu){return gtsam::noiseModel::Constrained::All(dim, mu);}, py::arg("dim"), py::arg("mu"));
}

//.def(py::init<const Key, const Key, const Key, const gtsam::Point2 ,
//                const gtsam::Point2, const gtsam::Cal3_S2 ,
//                const gtsam::SharedNoiseModel &>(),
//       py::arg("key1_pi"), py::arg("key2_pj"), py::arg("key3_di"),
//       py::arg("pixel_i"), py::arg("predicted_pixel_j"), py::arg("K"),
//       py::arg("noiseModel"));
