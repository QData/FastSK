cmake_minimum_required(VERSION 2.8.12)
project(fastsk)

add_subdirectory(pybind11)
pybind11_add_module(fastsk
	src/bindings.cpp
	src/kernel.cpp
	src/svm.cpp
	src/shared.cpp
	src/gakco_core.cpp
	src/dataset.cpp
	src/libsvm-code/libsvm.cpp
	src/libsvm-code/eval.cpp)