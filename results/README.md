# FastSK Experiments
How to replicate our results and generate the figures from our paper. 

## Setup
The string kernel baseline models are in the `baselines` folder. Unzip and build each of them before running these experiments. The `experiments.py` file assumes hard-coded paths to the executables for the baseline models, so don't move any of the files/executables in the `baselines` directory after building.

## Grid Search
To run a grid search over the hyperparameter space (g, m, and C) to find the optimal parameters, run:

## AUC Experiments

## AUC and Approx Algo Variance vs Number of Iterations
Running the DNA experiments (this will take a while):
```
python experiments.py --output-dir ./stdevs --stdev-I
```
Then recompile gkmsvm for protein and now run:
```
python experiments.py --output-dir ./stdevs --stdev-I --gkm-mode prot
```
Creating the plots:
```
```

## Impact of g parameter


## Timing Experiments
All timing results were completed on a server with the following specs:
-   12 Intel Core i7-6850K CPU @ 3.60GHz
-   15,360 KB cache
-   132 GB RAM

(for members of QData: the qcuda3 and 4 servers have these specs, as of January 2020)

### Multithreading Experiments

### AUC vs Time Figures

## Neural Network Baselines
### Character-Level CNN
### LSTM
