#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "svm.hpp"
#include "kernel.hpp"

#include <iostream>
#include <assert.h>

#include <string>

namespace py = pybind11;

PYBIND11_MODULE(fastsk, m) {
    // py::class_<SVM>(m, "SVM")
    //     .def(py::init<int, int, double, double, double, std::string>(), 
    //         py::arg("g"), 
    //         py::arg("m"), 
    //         py::arg("C")=1.0, 
    //         py::arg("nu")=0.5, 
    //         py::arg("eps")=0.001, 
    //         py::arg("kernel")="linear")
    //     .def("toString", 
    //         &SVM::toString)
    //     .def("fit", 
    //         &SVM::fit, 
    //         py::arg("train_file"), 
    //         py::arg("test_file"), 
    //         py::arg("dict")="", 
    //         py::arg("quiet")=false, 
    //         py::arg("kernel_file")="")
    //     .def("predict", 
    //         &SVM::predict, 
    //         py::arg("predictions_file"))
    //     .def("fit_numerical", &SVM::fit_numerical,
    //         py::arg("Xtrain"),
    //         py::arg("Ytrain"),
    //         py::arg("Xtest"),
    //         py::arg("Ytest"),
    //         py::arg("kernel_file")="")
    //     .def("cv", &SVM::cv,
    //         py::arg("X"),
    //         py::arg("Y"),
    //         py::arg("num_folds")=7)
    //     .def("fit_from_arrays", [](SVM &self, py::list Xtrain, 
    //             py::array_t<int> Ytrain, 
    //             py::list Xtest, py::array_t<int> Ytest,
    //             std::string kernel_file) {
    //         std::vector<std::string> xtrain_vec;
    //         for (py::handle seq_handle : Xtrain) {
    //             std::string seq_string = seq_handle.attr("__str__")().cast<std::string>();
    //             xtrain_vec.push_back(seq_string);
    //         }
    //         py::buffer_info info = Ytrain.request();
    //         if (info.ndim != 1) {
    //             throw std::runtime_error("Ytrain must be a 1-dimensional array");
    //         }
    //         int y_train_len = info.size;
    //         int n_str_train = xtrain_vec.size();
    //         assert(y_train_len == n_str_train);
    //         int* ptr = static_cast<int *>(info.ptr);
    //         std::vector<int> ytrain_vec(ptr, ptr + y_train_len);

    //         std::vector<std::string> xtest_vec;
    //         for (py::handle seq_handle : Xtest) {
    //             std::string seq_string = seq_handle.attr("__str__")().cast<std::string>();
    //             xtest_vec.push_back(seq_string);
    //         }
    //         py::buffer_info test_info = Ytest.request();
    //         if (test_info.ndim != 1) {
    //             throw std::runtime_error("Ytest must be a 1-dimensional array");
    //         }
    //         int y_test_len = test_info.size;
    //         int n_str_test = xtest_vec.size();
    //         assert(y_t == n_str_test);
    //         int* test_ptr = static_cast<int *>(test_info.ptr);
    //         std::vector<int> ytest_vec(test_ptr, test_ptr + y_test_len);

    //         return self.fit_from_arrays(xtrain_vec, ytrain_vec, xtest_vec, ytest_vec, kernel_file);
    //     })
    //     .def("score", 
    //         &SVM::score, 
    //         py::arg("metric")="accuracy");

    py::class_<Kernel>(m, "Kernel")
        .def(py::init<int, int, int, bool, double, int, bool>(), 
            py::arg("g"), 
            py::arg("m"),
            py::arg("t")=-1,
            py::arg("approx")=false,
            py::arg("delta")=0.025,
            py::arg("max_iters")=-1,
            py::arg("skip_variance")=false)
        .def("compute", &Kernel::compute,
            py::arg("Xtrain"),
            py::arg("Xtest"))
        .def("compute_train", &Kernel::compute_train,
            py::arg("Xtrain"))
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
