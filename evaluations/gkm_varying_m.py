import os
import os.path as osp
import sys
import argparse
import json
import numpy as np
import pandas as pd
import time
import subprocess

def get_args():
    parser = argparse.ArgumentParser(description='gkm Varying m Experiments')
    parser.add_argument('--datasets', type=str, required=True,
        help="Where to find the datasets")
    parser.add_argument('--prefix', type=str, required=True,
        help="Name of the dataset to use")
    parser.add_argument('--dict', type=str, required=False, 
        help='Dictionary file')
    parser.add_argument('--out', type=str, required=True, 
        help='Name of log file to save results')

    return parser.parse_args()

args = get_args()
file = args.out
datasets = args.datasets
prefix = args.prefix
result_file = args.out

if not osp.exists("./temp"):
    os.makedirs("./temp")
kernel_file = osp.join("./temp/kernel.out")

train_pos_file = osp.join(datasets, prefix + ".train.pos.fasta")
train_neg_file = osp.join(datasets, prefix + ".train.neg.fasta")

max_g = 15
for g in range(2, max_g + 1):
    m = g // 2
    k = g - m

    command = ["./gkmsvm_kernel",
        '-a', str(2),
        '-l', str(g), 
        '-k', str(k), 
        '-d', str(m),
        '-R']
    if args.dict is not None:
        command += ['-A', args.dict]
    command += [train_pos_file, train_neg_file, kernel_file]
    print(' '.join(command))
    start_time = time.time()
    output = subprocess.check_output(command)
    exec_time = time.time() - start_time
    result = {
        "dataset": prefix,
        "g": g,
        "m": m,
        "k": k,
        "time": exec_time,
    }
    with open(result_file, 'a+') as f:
        f.write(str(result) + '\n')
