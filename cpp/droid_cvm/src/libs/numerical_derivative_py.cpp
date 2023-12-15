#include <numerical_derivative_py.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(numerical_derivative_py, m) {

  m.doc() = "numerical derivative implementaion of different functions";
  m.def("numerical_derivative_dba", &numerical_derivative_dba);
}
