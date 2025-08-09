#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "hawkes_engine.h"
namespace py = pybind11;
PYBIND11_MODULE(pyhawkes, m) {
    py::class_<HawkesEngine>(m, "HawkesEngine")
        .def(py::init<>())
        .def("process_events", [](HawkesEngine& self, py::array_t<double> timestamps) {
            py::buffer_info buf = timestamps.request();
            return self.processEvents(static_cast<double*>(buf.ptr), buf.size);
        });
}
