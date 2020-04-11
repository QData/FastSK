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
**With pip**
```
pip install -i https://test.pypi.org/simple/ fastsk-test
```

**From source**

Clone this repository:
```
git clone --recursive https://github.com/QData/FastSK.git
```
Then run:

```
cd FastSK
pip install -r requirements.txt
pip install ./fastsk
```

or

```
pip3 install ./fastsk
```

**C++ Version**
Clone this repository:
```
git clone --recursive https://github.com/QData/FastSK.git
```
Then run:
```
cd FastSK
make
```
A `fastsk` executable will be installed to the `bin` directory, which you can use for kernel computation and inference. For example:
```
./bin/fastsk -g 10 -m 6 -C 1 -t 1 -a data/EP300.train.fasta data/EP300.test.fasta
```
This will run the approximate kernel algorithm on the EP300 TFBS dataset using a feature length of `g = 10` with up to `m = 6` mismatches. It will then train and evaluate an SVM classifier with the SVM parameter `C = 1`.

## Python Tutorial
Example:
```
from fastsk import FastSK
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

## Compute kernel matrix
fastsk = FastSK(g=10, m=6, t=1, approx=True)
fastsk.compute_kernel('data/EP300.train.fasta', 'data/EP300.test.fasta')

Xtrain = fastsk.get_train_kernel()
Xtest = fastsk.get_test_kernel()

## Use linear SVM
svm = LinearSVC(C=C)
clf = CalibratedClassifierCV(svm, cv=5).fit(Xtrain, Ytrain)

## Evaluate
acc = clf.score(Xtest, Ytest)
probs = clf.predict_proba(Xtest)[:,1]
auc = metrics.roc_auc_score(Ytest, probs)

print("Linear SVM:\n\tAcc = {}, AUC = {}".format(acc, auc))
```

## Special notes for Windows
**Compiler requirements**

Pybind11 requires a C++11 compliant compiler, i.e Visual Studio 2015 on Windows.
This applies to all Python versions, including 2.7. Unlike regular C extension
modules, it's perfectly fine to compile a pybind11 module with a VS version newer
than the target Python's VS version.

## License
See the LICENSE file.
