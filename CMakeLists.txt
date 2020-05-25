cmake_minimum_required(VERSION 2.8.12)
project(fastsk)

add_subdirectory(src/pybind11)
pybind11_add_module(_fastsk
    src/fastsk/bindings.cpp
    src/fastsk/fastsk.cpp
    src/fastsk/fastsk_kernel.cpp
    src/fastsk/shared.cpp
    src/fastsk/utils.cpp
    src/fastsk/libsvm-code/svm.cpp
    src/fastsk/libsvm-code/eval.cpp
)
