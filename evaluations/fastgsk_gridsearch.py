import os
import os.path as osp
import sys
sys.path.append('./fastsk')
import argparse
import json
import numpy as np
from fastsk import Kernel
from utils import FastaUtility, GkmRunner, GaKCoRunner, FastskRunner
import pandas as pd
import time
from scipy import special
import multiprocessing
import subprocess

def get_args():
    parser = argparse.ArgumentParser(description='iGakco Evaluations')
    parser.add_argument('--dataset', type=str)

    return parser.parse_args()

args = get_args()
dataset = args.dataset
log_file = dataset + '_gridsearch.out'

def evaluate_clf(clf, Xtest, Ytest):
    acc = clf.score(Xtest, Ytest)
    probs = clf.predict_proba(Xtest)[:,1]
    auc = metrics.roc_auc_score(Ytest, probs)
    return acc, auc

### Run gridsearch
def grid_search():
    best_auc, best_params = 0, {}
    min_g, max_g = 4, 15
    g_vals = list(range(min_g, max_g + 1))
    C_vals = [10 ** i for i in range(-3, 3)]
    for C in C_vals:
        for g in g_vals:
            for m in range(0, g - 2):
                k = g - m
                
                fastsk = FastskRunner(dataset)
                acc, auc = fastsk.train_and_test(g, m, t=1, approx=True, C=C)

                log = {
                    "dataset": dataset,
                    "C": C,
                    "g": g,
                    "k": k,
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
