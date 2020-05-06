# DNA and Protein Classification DNN Baseline Models

+ On DNA and protein sequences, we tried two DNN based sequence  classifier implemented with PyTorch. 

+ Originally created for comparison with [FastSK](https://github.com/QData/FastSK), a new string kernel support vector machine classifier for sequence annotation.

+ You need to have pytorch for running the above command. If you don't, please check [URL](https://pytorch.org/get-started/locally/)

+ run_charcnn_experiments.py is a script for running the charcnn on all 27 datasets.
+ run_cnn.py runs the charcnn on just one dataset

+ run_rnn.py runs the lstm and does a grid search on just one dataset