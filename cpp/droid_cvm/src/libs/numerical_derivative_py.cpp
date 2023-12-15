#include <numerical_derivative_py.h>
#include <boost/shared_ptr.hpp>

#include <pybind11/eigen.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
PYBIND11_DECLARE_HOLDER_TYPE(TYPE_PLACEHOLDER_DONOTUSE, boost::shared_ptr<TYPE_PLACEHOLDER_DONOTUSE>);

namespace PYBIND11_NAMESPACE { namespace detail {
    template <typename T>
    struct type_caster<boost::optional<T>> : optional_caster<boost::optional<T>> {};
}}
namespace py = pybind11;

PYBIND11_MODULE(numerical_derivative_py, m) {

  m.doc() = "numerical derivative implementaion of different functions";
  m.def("numerical_derivative_dba", &numerical_derivative_dba);
}
