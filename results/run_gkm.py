import os
import os.path as osp
import subprocess
import time
import sys
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score
import time

GKM_DIR = osp.join('baselines', 'gkmsvm')
GKM_KERNEL = osp.join(GKM_DIR, 'gkmsvm_kernel')
GKM_TRAIN = osp.join(GKM_DIR, 'gkmsvm_train')
GKM_CLASSIFY = osp.join(GKM_DIR, 'gkmsvm_classify')

def get_args():
    parser = argparse.ArgumentParser(description='Analyze gkm results data')
    parser.add_argument('--data_dir', type=str, default='./baselines/gkm_data', 
        help='Dataset directory', metavar='DATA_DIR')
    parser.add_argument('--prefix', type=str, required=True, 
        help='Dataset prefix', metavar='PREFIX')
    parser.add_argument('--outdir', type=str, default='./temp', metavar='NAME',
        help='Directory to store intermediate and output files')
    parser.add_argument('-g', type=int, required=True)
    parser.add_argument('-m', type=int, required=True)
    parser.add_argument('--dict', type=str, required=False, 
        help='Dictionary file name (not needed for DNA datasets)')

    return parser.parse_args()

def read_preds(file):
    preds = []
    with open(file, 'r') as f:
        for line in f:
            line = line.split()
            assert len(line) == 2
            preds.append(float(line[1]))
    return preds

def get_accuracy(pos_preds, neg_preds):
    accuracy = 0
    num_correct = 0
    num_pred = len(pos_preds) + len(neg_preds)
    for pred in pos_preds:
        if pred > 0:
            num_correct += 1
    for pred in neg_preds:
        if pred <= 0:
            num_correct += 1
    return num_correct / num_pred

def get_auc(pos_preds, neg_preds):
    ytrue = [1 for _ in pos_preds] + [-1 for _ in neg_preds]
    yscore = [score for score in pos_preds] + [score for score in neg_preds]
    auc = roc_auc_score(ytrue, yscore)
    return auc

args = get_args()
g, m = args.g, args.m
k = g - m
# input files
dir, prefix = args.data_dir, args.prefix
train_pos_file = osp.join(dir, prefix + '.train.pos.fasta')
train_neg_file = osp.join(dir, prefix + '.train.neg.fasta')
test_pos_file = osp.join(dir, prefix + '.test.pos.fasta')
test_neg_file = osp.join(dir, prefix + '.test.neg.fasta')
# output files
outdir = args.outdir
if not osp.exists(outdir):
    os.makedirs(outdir)
kernel_file = osp.join(outdir, prefix + '_kernel.out')
svm_file_prefix = osp.join(outdir, "svmtrain")
svmalpha = svm_file_prefix + '_svalpha.out'
svseq = svm_file_prefix + '_svseq.fa'
pos_pred_file = osp.join(outdir, prefix + '.preds.pos.out')
neg_pred_file = osp.join(outdir, prefix + '.preds.neg.out')

### compute kernel ###
print("Computing kernel...")
command = [GKM_KERNEL,
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

print("Train kernel time: ", exec_time)

### train SVM ###
print("Training model...")
command = [GKM_TRAIN, kernel_file, train_pos_file, 
    train_neg_file, svm_file_prefix]
print(' '.join(command))
output = subprocess.check_output(command)

### test ###
print("Getting predictions...")
# get pos preds
command = [GKM_CLASSIFY,
    "-l", str(g), 
    "-k", str(k), 
    "-d", str(m),
    '-R']
if args.dict is not None:
    command += ['-A', args.dict]
command += [test_pos_file, svseq, svmalpha, pos_pred_file]
print(' '.join(command))
subprocess.check_output(command)
# get neg preds
command = [GKM_CLASSIFY,
    "-l", str(g), 
    "-k", str(k), 
    "-d", str(m),
    '-R']
if args.dict is not None:
    command += ['-A', args.dict]

command += [test_neg_file, svseq, svmalpha, neg_pred_file]
print(' '.join(command))
subprocess.check_output(command)

### evaluate ###
pos_preds = read_preds(pos_pred_file)
neg_preds = read_preds(neg_pred_file)

print("Computing accuracy...")
accuracy = get_accuracy(pos_preds, neg_preds)
print("Computing AUC...")
auc = get_auc(pos_preds, neg_preds)
print("Accuracy = {}, AUC = {}".format(accuracy, auc))
