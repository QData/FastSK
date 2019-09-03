# DNA and Protein Classification Baseline Models
A DNA and protein sequence LSTM classifier implemented with PyTorch. Originally created for comparison with [iGakco-SVM](https://github.com/QData/iGakco-SVM), a new gapped k-mer support vector machine classifier for sequence annotation.

Usage:
```
python main.py --trn data/1.1.test.fasta --tst data/1.1.test.fasta --batch 64 --file results.txt
```
