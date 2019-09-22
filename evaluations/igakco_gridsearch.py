'''Grid search for finding best
parameters to optimize AUC using
the iGakco kernel
'''

import os
import os.path as osp
import sys
import argparse
import json
import numpy as np
from sklearn import metrics

'''For a linear kernel, we need the training and
testing data ahead of time. With igakco, we can use
an empirical kernel map (EKM) as the features to a linear
classify.
'''
RESULTS_DIR = './results'
if not osp.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def get_args():
    parser = argparse.ArgumentParser(description='iGakco Evaluations')
    parser.add_argument('--trn', type=str, 
        required=True, help='Training file', metavar='1.1.train.fasta')
    parser.add_argument('--tst', type=str, 
        required=True, help='Test file', metavar='1.1.test.fasta')
    parser.add_argument('--file', type=str, 
        required=True, help='Name of log file to save grid search results')

    return parser.parse_args()

args = get_args()
train_file, test_file = args.trn, args.tst
log_file = args.file
log_file = osp.join(RESULTS_DIR, log_file)

### Read the data
reader = FastaUtility()
Xtrain_raw, Ytrain = reader.read_data(train_file)
Xtest_raw, Ytest = reader.read_data(test_file)
Ytest = np.array(Ytest).reshape(-1, 1)

def evaluate_clf(clf, Xtest, Ytest):
    acc = clf.score(Xtest, Ytest)
    probs = clf.predict_proba(Xtest)[:,1]
    auc = metrics.roc_auc_score(Ytest, probs)
    return acc, auc

### Run gridsearch
def grid_search():
    best_auc, best_params = 0, {}
    min_g, max_g = 5, 15
    g_vals = list(range(min_g, max_g + 1))
    C_vals = [10 ** i for i in range(-3, 3)]
    for C in C_vals:
        for g in g_vals:
            for m in range(1, g):
                ### Compute the igakco kernel
                kernel = Kernel(g=g, m=m)
                kernel.compute(Xtrain_raw, Xtest_raw)
                Xtrain = kernel.train_kernel()
                Xtest = kernel.test_kernel()

                ### Use linear SVM
                svm = LinearSVC(C=C)
                clf = CalibratedClassifierCV(svm, cv=5).fit(Xtrain, Ytrain)
                acc, auc = evaluate_clf(clf, Xtest, Ytest)

                ### Save Results
                log = {
                    "train": train_file,
                    "test": test_file,
                    "C": C,
                    "g": g,
                    "m": m,
                    "acc": acc,
                    "auc": auc
                }
                print(log)
                if auc > best_auc:
                    best_auc = auc
                    best_params = log

                with open(log_file, 'a+') as f:
                    f.write(str(log) + '\n')

    return best_params

best_params = grid_search()
with open(log_file, 'a+') as f:
    f.write("Best params:\n" + str(best_params) + '\n')
