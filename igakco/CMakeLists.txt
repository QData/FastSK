cmake_minimum_required(VERSION 2.8.12)
project(igakco)

add_subdirectory(pybind11)
pybind11_add_module(igakco
	src/bindings.cpp 
	src/svm.cpp
	src/shared.cpp
	src/gakco_core.cpp
	src/dataset.cpp
	src/libsvm-code/libsvm.cpp
	src/libsvm-code/eval.cpp)
