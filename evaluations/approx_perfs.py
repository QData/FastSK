import os
import os.path as osp
import sys
sys.path.append('./fastsk')
import argparse
import json
import numpy as np
from fastsk import Kernel
from utils import FastaUtility
import pandas as pd
import time
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn import metrics

def evaluate_clf(clf, Xtest, Ytest):
    acc = clf.score(Xtest, Ytest)
    probs = clf.predict_proba(Xtest)[:,1]
    auc = metrics.roc_auc_score(Ytest, probs)
    return acc, auc

def get_args():
    parser = argparse.ArgumentParser(description='FastSK')
    parser.add_argument('--datasets', type=str, required=True,
        help="Where to find the datasets")
    parser.add_argument('--params', type=str, required=True,
        help="The paramters file")
    parser.add_argument('--out', type=str, required=True, 
        help='Name of log file to save timing results')

    return parser.parse_args()

args = get_args()
outfile = args.out
datasets = args.datasets
parameters = args.params

df = pd.read_csv(parameters)
params = df.to_dict('records')

def train_kernel_time(g, m, t, approx, I=50, Xtrain):
    start = time.time()
    kernel = Kernel(g=g, m=m, t=t, approx=approx, I=I)
    kernel.compute_train(Xtrain)
    end = time.time()

    return end - start

results = {
    'Dataset' : [],
    'g' : [],
    'm' : [],
    'k' : [],
    'C' : [],
    'acc' : [],
    'auc' : [],
    'threads' : [],
    'approx' : [],
    'train_k_time' : [],
}

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

    ktime = train_kernel_time(g, m, t=1, approx=True, I=50, Xtrain=Xtrain)

    # ### train-test kernel
    kernel = Kernel(g=g, m=m, t=1, approx=True, I=50)
    kernel.compute(Xtrain, Xtest)
    Xtrain = kernel.train_kernel()
    Xtest = kernel.test_kernel()

    # ### Use linear SVM
    svm = LinearSVC(C=C)
    clf = CalibratedClassifierCV(svm, cv=5).fit(Xtrain, Ytrain)
    acc, auc = evaluate_clf(clf, Xtest, Ytest)
    print("Ktime, Acc, AUC = {}, {}, {}".format(ktime, acc, auc))

    results['Dataset'].append(p['Dataset'])
    results['g'].append(g)
    results['m'].append(m)
    results['k'].append(k)
    results['C'].append(C)
    results['I'].append(50)
    results['acc'].append(acc)
    results['auc'].append(auc)
    results['threads'].append(1)
    results['approx'].append(True)
    results['train_k_time'].append(ktime)

    res = pd.DataFrame(results)
    res.to_csv(outfile, index=False)
