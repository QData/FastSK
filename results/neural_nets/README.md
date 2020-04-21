# DNA and Protein Classification DNN Baseline Models
A DNA and protein sequence LSTM classifier implemented with PyTorch. Originally created for comparison with [FastSK](https://github.com/QData/FastSK), a new string kernel support vector machine classifier for sequence annotation.

Usage:
```
python main.py --trn data/1.1.test.fasta --tst data/1.1.test.fasta --batch 64 --file results.txt
```

+ You need to have pytorch for running the above command. If you don't, please check out [URL](https://pytorch.org/get-started/locally/)