+ You do need to have pytorch installed for running the codes in this folder. If you don't, please run the following or check  [URL](https://pytorch.org/get-started/locally/)
```
pip install torch torchvision 
```

# We compare FastSK to DNN Baseline Models

+ we tried two DNN based sequence  classifier implemented with PyTorch as DNN baselines 

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

+ one more possible usage: (on each CNN hyperparameter configration, we repeat 5-random-seeded runs and report average performance numbers)
```
cd neural_nets
python cnn_hyperTrTune.py --trn ../../data/1.1.train.fasta --tst ../../data/1.1.test.fasta --trn_size 1.0 --lr 0.01 --datasetTag 1.1 --opt_mtd sgd --epochs 20
```

## Some experimental results comparing FastSK and CharCNN across multiple datasets when varying training size 

<img src="trainsize_varyresults/dna.png?raw=true" width="800">

<img src="trainsize_varyresults/protein.png" width="800">

<img src="trainsize_varyresults/nlp.png" width="800">
