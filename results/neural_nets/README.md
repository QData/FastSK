# DNA and Protein Classification DNN Baseline Models

+ On DNA and protein sequences, we tried two DNN based sequence  classifier implemented with PyTorch. 

+ Originally created for comparison with [FastSK](https://github.com/QData/FastSK), a new string kernel support vector machine classifier for sequence annotation.

+ You need to have pytorch for running the above command. If you don't, please check [URL](https://pytorch.org/get-started/locally/)

+ run_cnn_allData.py is a script for running the charcnn on all 27 datasets.
+ run_cnn.py runs the charcnn on just one dataset
+ run_rnn.py runs the lstm  on just one dataset

+ For instance, one usage:
```
cd neural_nets
python run_cnn_allData.py 
```

+ For instance, another usage:
```
cd neural_nets
python run_cnn_hyperTrTune.py 
```

## Experimental results comparing FastSK and CharCNN across multiple datasets when varying training size 

<img src="trainsize_varyresults/dna.png?raw=true" width="800">

<img src="trainsize_varyresults/protein.png" width="800">

<img src="trainsize_varyresults/nlp.png" width="800">
