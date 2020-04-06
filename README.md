# FastSK: Fast and Accurate Sequence Classification using Support Vector Machines

[![Build Status](https://travis-ci.org/pybind/fastsk.svg?branch=master)](https://travis-ci.org/pybind/fastsk)
[![Build status](https://ci.appveyor.com/api/projects/status/57nnxfm4subeug43/branch/master?svg=true)](https://ci.appveyor.com/project/dean0x7d/cmake-example/branch/master)

A Python package and string kernel algorithm for training SVM classifiers for sequence analysis. Built with the FastSK gapped k-mer algorithm, [pybind11](https://github.com/pybind/pybind11), and [LIBSVM](https://github.com/cjlin1/libsvm).


## Prerequisites

**On Unix (Linux, OS X)**

* A compiler with C++11 support
* CMake >= 2.8.12

**On Windows**

* Visual Studio 2015 (required for all Python versions, see notes below)
* CMake >= 3.1


## Installation (Linux and MacOS)

**From source**

Clone this repository:
```
git clone --recursive https://github.com/QData/FastSK.git
```
The `--recursive` flag is to make sure the Pybind11 library is cloned as well. Then run:

```
cd FastSK
pip install ./fastsk
```

or

```
pip3 install ./fastsk
```

## Tutorial
Example usage:
```
from fastsk import SVM
svm = SVM(g=7, m=2, C=0.7)
svm.fit(train_file="1.1.train.fasta", test_file="1.1.test.fasta", quiet=False, kernel_file="output.txt")
svm.predict("predictions.txt")
```
This will use the provided parameters to build train and test kernel matrices and train an SVM classifier. The `predict` call will write the predicted labels of the provided `test_file` to `predictions.txt`.

Alternatively, we can train by feeding in arrays of sequences and labels:
```
from fastsk import SVM

xtrain = ["ACACA", "AAACA"]
ytrain = [1, 0]
xtest = ["AAAAA", "ACACA"]
ytest = [1, 0]

svm = SVM(g=3, m=2, C=0.7)
svm.fit_from_arrays(xtrain, ytrain, xtest, ytest, "kernel.txt")
svm.predict("preds.txt")
```

## Documentation
Constructor:
* `g` (required)
* `m` (required)
* `C` (optional, default=1.0)
* `nu` (optional, default=0.5)
* `eps` (optional, default = 0.001) - LIBSVM epsilon parameter
* `kernel` (optional, default = 'linear'). Options: linear, fastsk, rbf

Fit:
* `train_file` (required)
* `test_file` (required)
* `dict` (optional). A dictionary file for the sequences in the train and test files. Default behavior is to infer the dictionary from the files.
* `quiet` (optional, default=false). Whether to be verbose.
* `kernel_file` (optional). If provided, the kernel matrix will be printed to the provided file. Otherwise, kernel matrix will not be saved.

Predict:
* `predictions_file` (required). File where predictions will be written. Format is one prediction, a single number, per line.

## Special notes for Windows
**Compiler requirements**

Pybind11 requires a C++11 compliant compiler, i.e Visual Studio 2015 on Windows.
This applies to all Python versions, including 2.7. Unlike regular C extension
modules, it's perfectly fine to compile a pybind11 module with a VS version newer
than the target Python's VS version.

## License
See the LICENSE file.
