# FastSK: fast sequence analysis with gapped string kernels (Fast-GKM-SVM)


This Github repo provides improved algorithms for implementing gkm-svm string kernel calculations. We provide C++ version of the algorithm implementation and a python wrapper (making to a python package) for the C++ implementation. Our package provides fast and accuate gkm-svm based training SVM classifiers and regressors for gkm string kernel based sequence analysis. 

This Github is built with a novel and fast algorithm design for implementing gapped k-mer algorithm, [pybind11](https://github.com/pybind/pybind11), and [LIBSVM](https://github.com/cjlin1/libsvm).

#### More details of algorithms and results now in: [Bioinformatics 2020](https://academic.oup.com/bioinformatics/article/36/Supplement_2/i857/6055916)

## Prerequisites

**On Windows**

* Visual Studio 2015 (required for all Python versions, see notes below)
* CMake >= 3.1


## Installation via Pip Install (Linux and MacOS)

Clone this repository and run:

```bash
git clone --recursive https://github.com/QData/FastSK.git
cd FastSK
pip install -r requirements.txt
pip install .
```

#### The pip intallation of FastSK has been tested successfully on CentOS, Red Hat, MacOS and WindowsXP.


## Python Version Tutorial

### Example Jupyter notebook  
- 'docs/2demo/fastDemo.ipynb'


### Example python usage script:
```
cd test
python run_check.py 
```

### You can check if fastsk library is installed correctly in python shell:

```
from fastsk import FastSK

## Compute kernel matrix
fastsk = FastSK(g=10, m=6, t=1, approx=True)
```

### Experimental Results, Baselines, Utility Codes and Setup

- We have provided all datasets we used in the subfolder "data"
- We have provided all scripts we used to generate results under the subfolder "results"

#### Grid Search for FastSK and gkm-svm baseline
To run a grid search over the hyperparameter space (g, m, and C) to find the optimal parameters, e.g, one utility code:
```
cd results/
python run_gridsearch.py
```

#### When comparing with Deep Learning baselines
+ You do need to have pytorch installed
```
pip install torch torchvision
```
+ One utility code: on all datasets with hyperparameter tuning of charCNN and each configure with 5 random-seeding repeats:
```
cd results/neural_nets
python run_cnn_hyperTrTune.py 
```
+ We have many other utility codes helping users to run CNN and RNN baselines

#### Some of our exprimental results comparing FastSK with baselines wrt performance and speed


<img src="results/spreadsheets/Figure5.png" width="800">

<img src="results/spreadsheets/Table1.png" width="800">

<img src="results/spreadsheets/Table2.png" width="800">


#### Some of our exprimental results comparing FastSK with Character based Convolutional Neural Nets (CharCNN) when varying training size. 

<img src="results/neural_nets/trainsize_varyresults/dna.png" width="800">

<img src="results/neural_nets/trainsize_varyresults/protein.png" width="800">

<img src="results/neural_nets/trainsize_varyresults/nlp.png" width="800">


#### To Do: 
* a detailed user document, with example input files, output files, code, and perhaps a user group where people can post their questions


### Citations

If you find this tool useful, please cite us!

```
@article{fast-gkm-svm,
    author = {Blakely, Derrick and Collins, Eamon and Singh, Ritambhara and Norton, Andrew and Lanchantin, Jack and Qi, Yanjun},
    title = "{FastSK: fast sequence analysis with gapped string kernels}",
    journal = {Bioinformatics},
    volume = {36},
    number = {Supplement_2},
    pages = {i857-i865},
    year = {2020},
    month = {12},
    abstract = "{Gapped k-mer kernels with support vector machines (gkm-SVMs) have achieved strong predictive performance on regulatory DNA sequences on modestly sized training sets. However, existing gkm-SVM algorithms suffer from slow kernel computation time, as they depend exponentially on the sub-sequence feature length, number of mismatch positions, and the task’s alphabet size.In this work, we introduce a fast and scalable algorithm for calculating gapped k-mer string kernels. Our method, named FastSK, uses a simplified kernel formulation that decomposes the kernel calculation into a set of independent counting operations over the possible mismatch positions. This simplified decomposition allows us to devise a fast Monte Carlo approximation that rapidly converges. FastSK can scale to much greater feature lengths, allows us to consider more mismatches, and is performant on a variety of sequence analysis tasks. On multiple DNA transcription factor binding site prediction datasets, FastSK consistently matches or outperforms the state-of-the-art gkmSVM-2.0 algorithms in area under the ROC curve, while achieving average speedups in kernel computation of ∼100× and speedups of ∼800× for large feature lengths. We further show that FastSK outperforms character-level recurrent and convolutional neural networks while achieving low variance. We then extend FastSK to 7 English-language medical named entity recognition datasets and 10 protein remote homology detection datasets. FastSK consistently matches or outperforms these baselines.Our algorithm is available as a Python package and as C++ source code at https://github.com/QData/FastSKSupplementary data are available at Bioinformatics online.}",
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btaa817},
    url = {https://doi.org/10.1093/bioinformatics/btaa817},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/36/Supplement\_2/i857/35337038/btaa817.pdf},
}
```




### Legacy: If you prefer using the executable made from the Pure C++ source code (without python wrapper or R wrapper)

- you can clone this repository:
```
git clone --recursive https://github.com/QData/FastSK.git
```
then run
```
cd FastSK
make
```
A `fastsk` executable will be installed to the `bin` directory, which you can use for kernel computation and inference. For example:
```
./bin/fastsk -g 10 -m 6 -C 1 -t 1 -a data/EP300.train.fasta data/EP300.test.fasta
```
This will run the approximate kernel algorithm on the EP300 TFBS dataset using a feature length of `g = 10` with up to `m = 6` mismatches. It will then train and evaluate an SVM classifier with the SVM parameter `C = 1`.
