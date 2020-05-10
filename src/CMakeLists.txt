cmake_minimum_required(VERSION 2.8.12)
project(fastsk)

add_subdirectory(pybind11)
pybind11_add_module(fastsk
    fastsk/bindings.cpp
    fastsk/fastsk.cpp
    fastsk/fastsk_kernel.cpp
    fastsk/shared.cpp
    fastsk/utils.cpp
    fastsk/libsvm-code/svm.cpp
    fastsk/libsvm-code/eval.cpp
)
