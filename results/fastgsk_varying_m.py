import os
import os.path as osp
import sys
sys.path.append('./fastsk')
import argparse
import json
import numpy as np
from fastsk import Kernel
from utils import FastaUtility
import time

RESULTS_DIR = './results'
if not osp.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def get_args():
    parser = argparse.ArgumentParser(description='fastsk Timing Experiments')
    parser.add_argument('--datasets', type=str, required=True,
        help="Where to find the datasets")
    parser.add_argument('--prefix', type=str, required=True,
        help="Which dataset to use")
    parser.add_argument('--out', type=str, required=True, 
        help='Name of log file to save timing results')
    parser.add_argument('-t', '--threads', type=int, required=False,
        default=20, help='Name of log file to save timing results')

    return parser.parse_args()

args = get_args()
file = args.out
datasets = args.datasets
prefix = args.prefix
results_file = osp.join(RESULTS_DIR, file)

train_file = prefix + '.train.fasta'
test_file = prefix + '.test.fasta'

train_file = osp.join(datasets, train_file)
test_file = osp.join(datasets, test_file)

g = 14
max_m = g - 1
for m in range(1, g):
    ### Read the data
    reader = FastaUtility()
    Xtrain, Ytrain = reader.read_data(train_file)
    Xtest, Ytest = reader.read_data(test_file)
    Ytest = np.array(Ytest).reshape(-1, 1)
    
    ### Compute the fastsk kernel
    kernel = Kernel(g=g, m=m, t=args.threads, approx=True, epsilon=0.9)
    start_time = time.time()
    kernel.compute_train(Xtrain)
    exec_time = time.time() - start_time

    result = {
        "g": g,
        "m": m,
        "k": g - m,
        "data": prefix,
        "time": exec_time,
        'threads': args.threads
    }

    with open(results_file, 'a+') as f:
        f.write(str(result) + '\n')
