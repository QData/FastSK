cmake_minimum_required(VERSION 2.8.12)
project(fastsk)

add_subdirectory(pybind11)
pybind11_add_module(_fastsk
    src/fastsk/_fastsk/bindings.cpp
    src/fastsk/_fastsk/fastsk.cpp
    src/fastsk/_fastsk/fastsk_kernel.cpp
    src/fastsk/_fastsk/shared.cpp
    src/fastsk/_fastsk/libsvm-code/svm.cpp
    src/fastsk/_fastsk/libsvm-code/eval.cpp
)
