#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "kernel.hpp"
#include <iostream>
#include <assert.h>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace std;

PYBIND11_MODULE(fastsk, m) {
    py::class_<Kernel>(m, "Kernel")
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
            (void (Kernel::*)(const string, const string)) &Kernel::compute_kernel,
            py::arg("Xtrain"),
            py::arg("Xtest")
        )
        .def("compute_kernel",
            (void (Kernel::*)(const string, const string, const string)) &Kernel::compute_kernel,
            py::arg("Xtrain"),
            py::arg("Xtest"),
            py::arg("dictionary_file")
        )
        .def("compute_kernel",
            (void (Kernel::*)(vector<string>, vector<string>)) &Kernel::compute_kernel,
            py::arg("Xtrain"),
            py::arg("Xtest")
        )
        .def("compute_kernel",
            (void (Kernel::*)(vector<vector<int> >, vector<vector<int> >)) &Kernel::compute_kernel,
            py::arg("Xtrain"),
            py::arg("Xtest")
        )
        .def("compute_train", 
            &Kernel::compute_train,
            py::arg("Xtrain")
        )
        .def("train_kernel", &Kernel::train_kernel)
        .def("test_kernel", &Kernel::test_kernel)
        .def("stdevs", &Kernel::get_stdevs)
        .def("save_kernel", &Kernel::save_kernel);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
