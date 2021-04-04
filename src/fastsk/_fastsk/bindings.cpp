#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>
#include "fastsk.hpp"
#include <string>
#include <vector>

namespace py = pybind11;
using namespace std;


PYBIND11_MODULE(_fastsk, m) {
    py::class_<FastSK>(m, "FastSK")
        .def(py::init<int, int, int, bool, double, int, bool>(), 
            py::arg("g"), 
            py::arg("m"),
            py::arg("t")=-1,
            py::arg("approx")=false,
            py::arg("delta")=0.025,
            py::arg("max_iters")=-1,
            py::arg("skip_variance")=false
        )
        .def("compute_kernel",
            (void (FastSK::*)(vector<vector<int> >, vector<vector<int> >)) &FastSK::compute_kernel,
            py::arg("Xtrain"),
            py::arg("Xtest")
        )
        .def("compute_train", 
            &FastSK::compute_train,
            py::arg("Xtrain")
        )
        .def("get_train_kernel", &FastSK::get_train_kernel)
        .def("get_test_kernel", &FastSK::get_test_kernel)
        .def("get_stdevs", &FastSK::get_stdevs)
        .def("save_kernel", &FastSK::save_kernel)
        .def("fit", &FastSK::fit,
            py::arg("C")=1.0,
            py::arg("nu")=0.5,
            py::arg("eps")=0.001,
            py::arg("kernel_type")="linear"
        )
        .def("score", &FastSK::score,
            py::arg("metric")="auc"
        );

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
