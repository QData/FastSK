import os
import os.path as osp
import sys
import argparse
import json
import numpy as np
import pandas as pd
import time
import subprocess

RESULTS_DIR = './results'
if not osp.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def get_args():
    parser = argparse.ArgumentParser(description='gkm Protein Experiments')
    parser.add_argument('--datasets', type=str, required=True,
        help="Where to find the datasets")
    parser.add_argument('--params', type=str, required=True,
        help="The paramters file")
    parser.add_argument('--dict', type=str, required=True,
        help="The protein dictionary file"),
    parser.add_argument('--out', type=str, required=True, 
        help='Name of log file to save results')

    return parser.parse_args()

args = get_args()
file = args.out
datasets = args.datasets
parameters = args.params
results_file = osp.join(RESULTS_DIR, file)
dict_file = args.dict

df = pd.read_csv(parameters)
experiments = df.to_dict('records')

for e in experiments:
    train_file = str(e['Dataset']) + '.train.fasta'
    test_file = str(e['Dataset']) + '.test.fasta'
    train_file = osp.join(datasets, train_file)
    test_file = osp.join(datasets, test_file)

    command = ["python", "run_gkm.py", 
        "--dir", datasets, 
        "--prefix", str(e["Dataset"]),
        "--outdir", "temp",
        "-g", str(e["g"]),
        "-m", str(e["m"]),
        "--dict", dict_file,
        "--results", args.out]
    print(' '.join(command))
    output = subprocess.check_output(command)
