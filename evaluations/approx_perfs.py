import os
import os.path as osp
import sys
sys.path.append('./igakco')
import argparse
import json
import numpy as np
from igakco import Kernel
from utils import FastaUtility
import pandas as pd
import time
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn import metrics

RESULTS_DIR = './results'
if not osp.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def evaluate_clf(clf, Xtest, Ytest):
    acc = clf.score(Xtest, Ytest)
    probs = clf.predict_proba(Xtest)[:,1]
    auc = metrics.roc_auc_score(Ytest, probs)
    return acc, auc

def get_args():
    parser = argparse.ArgumentParser(description='iGakco Timing Experiments')
    parser.add_argument('--datasets', type=str, required=True,
        help="Where to find the datasets")
    parser.add_argument('--params', type=str, required=True,
        help="The paramters file")
    parser.add_argument('--out', type=str, required=True, 
        help='Name of log file to save timing results')

    return parser.parse_args()

args = get_args()
file = args.out
datasets = args.datasets
parameters = args.params
results_file = osp.join(RESULTS_DIR, file)

df = pd.read_csv(parameters)
params = df.to_dict('records')

for p in params:
    print(p)
    train_file = p['Dataset'] + '.train.fasta'
    test_file = p['Dataset'] + '.test.fasta'
    train_file = osp.join(datasets, train_file)
    test_file = osp.join(datasets, test_file)

    g, m, k, C = p['g'], p['m'], p['k'], p['C']
    assert k == g - m
    
    ### Read the data
    reader = FastaUtility()
    Xtrain, Ytrain = reader.read_data(train_file)
    Xtest, Ytest = reader.read_data(test_file)
    Ytest = np.array(Ytest).reshape(-1, 1)

    ### Compute the igakco kernel
    kernel = Kernel(g=g, m=m, approx=True, epsilon=0.9)
    kernel.compute(Xtrain, Xtest)
    Xtrain = kernel.train_kernel()
    Xtest = kernel.test_kernel()

    ### Use linear SVM
    svm = LinearSVC(C=C)
    clf = CalibratedClassifierCV(svm, cv=5).fit(Xtrain, Ytrain)
    acc, auc = evaluate_clf(clf, Xtest, Ytest)
    print("Acc, AUC = {}, {}".format(acc, auc))

    result = {
        "data": p['Dataset'],
        "g": g,
        "m": m,
        "k": k,
        "C": C,
        "epsilon": 0.1,
        "acc": acc,
        "auc": auc
    }

    with open(results_file, 'a+') as f:
        f.write(str(result) + '\n')
