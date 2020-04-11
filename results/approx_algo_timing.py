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

RESULTS_DIR = './results'
if not osp.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def get_args():
    parser = argparse.ArgumentParser(description='fastsk Timing Experiments')
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
    train_file = p['Dataset'] + '.train.fasta'
    test_file = p['Dataset'] + '.test.fasta'
    train_file = osp.join(datasets, train_file)
    test_file = osp.join(datasets, test_file)

    g, m, k = p['g'], p['m'], p['k']
    assert k == g - m
    ### Read the data
    reader = FastaUtility()
    Xtrain, Ytrain = reader.read_data(train_file)
    Xtest, Ytest = reader.read_data(test_file)
    Ytest = np.array(Ytest).reshape(-1, 1)
    ### Compute the fastsk kernel
    kernel = Kernel(g=g, m=m, approx=True, epsilon=0.9)
    start_time = time.time()
    kernel.compute_train(Xtrain)
    exec_time = time.time() - start_time
    p['time'] = exec_time
    with open(results_file, 'a+') as f:
        f.write(str(p) + '\n')
