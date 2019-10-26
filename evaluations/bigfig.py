import os
import os.path as osp
import sys
sys.path.append('./fastsk')
import argparse
import json
import numpy as np
from fastsk import Kernel
from utils import FastaUtility, GkmRunner, FastskRunner
import pandas as pd
import time
from scipy import special

def time_fastsk(g, m, t, prefix, approx=False, max_iters=None):
    reader = FastaUtility()
    train_file = osp.join('./data/' + prefix + '.train.fasta')
    Xtrain, Ytrain = reader.read_data(train_file)

    start = time.time()
    if max_iters:
        kernel = Kernel(g=g, m=m, t=t, approx=approx, max_iters=max_iters)
    else:
        kernel = Kernel(g=g, m=m, t=t, approx=approx)
    kernel.compute_train(Xtrain)
    end = time.time()

    return end - start

def time_gkm(g, m, t, prefix):
    gkm_data = '/localtmp/dcb7xz/FastSK/baselines/gkm_data'
    gkm_exec = '/localtmp/dcb7xz/FastSK/baselines/gkmsvm'
    gkm = GkmRunner(gkm_exec, gkm_data, prefix, './temp')

    start = time.time()
    gkm.compute_kernel(g=g, m=m, t=t)
    end = time.time()

    return end - start

def time_gakco(t):
    pass

def time_blended(t):
    pass

def thread_experiment(dataset, g, m, k):
    output_csv = dataset + '_vary_threads.csv'
    results = {
        'fastsk_exact_time' : [],
        'fastsk_approx_time' : [],
        'fastsk_approx_time_t1' : [],
        'gkm_time' : [],
    }

    train_file = osp.join('./data', dataset + '.train.fasta')
    reader = FastaUtility()
    Xtrain, Ytrain = reader.read_data(train_file)

    for t in range(1, 21):
        fastsk_exact = time_fastsk(g, m, t, prefix=dataset, approx=False)
        fastsk_approx = time_fastsk(g, m, t, prefix=dataset, approx=True)
        fastsk_approx_t1 = time_fastsk(g, m, t=1, prefix=dataset, approx=True)
        gkm = time_gkm(g, m, t, prefix=dataset)

        results['fastsk_exact_time'].append(fastsk_exact)
        results['fastsk_approx_time'].append(fastsk_approx)
        results['fastsk_approx_time_t1'].append(fastsk_approx_t1)
        results['gkm_time'].append(gkm)

        log_str = "{} - exact: {}, approx: {} approx_t1: {}, gkm: {}"
        print(log_str.format(dataset, fastsk_exact, fastsk_approx, fastsk_approx_t1, gkm))

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

def run_thread_experiments(params):
    for p in params:
        dataset, type_, g, m, k = p['Dataset'], p['type'], p['g'], p['m'], p['k']
        assert k == g - m
        if type_ == 'dna':
            thread_experiment(dataset, g, m, k)

def g_experiment(dataset):
    output_csv = dataset + '_vary_g.csv'
    results = {
        'g': [],
        'm': [],
        'fastsk_exact_time' : [],
        'fastsk_approx_time' : [],
        'fastsk_approx_time_t1' : [],
        'gkm_time' : [],
    }
    train_file = osp.join('./data', dataset + '.train.fasta')
    reader = FastaUtility()
    Xtrain, Ytrain = reader.read_data(train_file)
    
    min_g, max_g = 4, 20
    for g in range(min_g, max_g + 1):
        m = g // 2

        fastsk_exact = time_fastsk(g, m, t=20, prefix=dataset, approx=False)
        fastsk_approx = time_fastsk(g, m, t=20, prefix=dataset, approx=True)
        fastsk_approx_t1 = time_fastsk(g, m, t=1, prefix=dataset, approx=True)
        gkm = time_gkm(g, m, t=20, prefix=dataset)

        results['g'].append(g)
        results['m'].append(m)
        results['fastsk_exact_time'].append(fastsk_exact)
        results['fastsk_approx_time'].append(fastsk_approx)
        results['fastsk_approx_time_t1'].append(fastsk_approx_t1)
        results['gkm_time'].append(gkm)

        log_str = "{} - exact: {}, approx: {} approx_t1: {}, gkm: {}"
        print(log_str.format(dataset, fastsk_exact, fastsk_approx, fastsk_approx_t1, gkm))

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

def run_g_experiments(params):
    for p in params:
        dataset, type_ = p['Dataset'], p['type']
        if type_ == 'dna':
            g_experiment(dataset)

def I_experiment(dataset, g, m, k, C):
    output_csv = dataset + '_vary_I.csv'
    results = {
        'I': [],
        'acc' : [],
        'auc' : [],
    }

    max_I = min(int(special.comb(g, m)), 500)
    iter_vals = list(range(10, max_I + 10, 10))
    for I in iter_vals:
        fastsk = FastskRunner(dataset)
        acc, auc = fastsk.train_and_test(g, m, t=1, approx=True, I=I, delta=0.025, C=C)
        log_str = "{}: I = {}, auc = {}, acc = {}".format(dataset, I, auc, acc)
        print(log_str)
        results['I'].append(I)
        results['acc'].append(acc)
        results['auc'].append(auc)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

def run_I_experiments(params):
    for p in params:
        dataset, type_, g, m, k, C = p['Dataset'], p['type'], p['g'], p['m'], p['k'], p['C']
        if dataset in ['ZZZ3', 'KAT2B', 'EP300_47848']:
            continue
        assert k == g - m
        I_experiment(dataset, g, m, k, C)


df = pd.read_csv('./evaluations/datasets_to_use.csv')
params = df.to_dict('records')

## Thread experiments
# run_thread_experiments(params)

## g experiments
#run_g_experiments(params)

## m experiments
pass

## AUC vs I experiments
#I_experiment('1.1', 8, 4, 4, 0.01)
run_I_experiments(params)

## AUC vs delta experiments
pass

## AUC vs g experiments
pass


