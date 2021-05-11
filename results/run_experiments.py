"""experiments.py
"""

__author__ = "Derrick Blakely"
__email__ = "dcb7xz@virginia.edu"
__date__ = "December 2019"

import os
import os.path as osp
import sys

sys.path.append("./fastsk")
import argparse
import json
import numpy as np
from fastsk import FastSK
from utils import *
import pandas as pd
import time
from scipy import special
from scipy.stats import sem, t
from scipy import mean
import multiprocessing
import subprocess
from tqdm import tqdm

# Default subprocess timeout (s)
TIMEOUT = 3600
MAXTIME = 1800

# Default locations for finding baseline programs
GKM_DATA = "./baselines/gkm_data"
GKM_EXEC = "./baselines/gkmsvm"
LSGKM_DATA = "./baselines/gkm_data"
LSGKM_EXEC = "./baselines/lsgkm-master/src"
FASTSK_DATA = "../data/"
BLENDED_EXEC = "./baselines/String_Kernel_Package/code/"
PROT_DICT = "../data/full_prot.dict.txt"
GKM_PROT_DICT = "./baselines/gkm_data/protein.dictionary.txt"


def get_args():
    parser = argparse.ArgumentParser(description="FastSK Experiments")
    parser.add_argument(
        "--threads",
        action="store_true",
        default=False,
        help="Run time vs number of threads experiments",
    )
    parser.add_argument(
        "--m-time", action="store_true", default=False, help="Run kernel time vs g"
    )
    parser.add_argument(
        "--g-time",
        action="store_true",
        default=False,
        help="Run kernel time vs g experiments",
    )
    parser.add_argument(
        "--g-time-full",
        action="store_true",
        default=False,
        help="Run full train and test time vs g experiments",
    )
    parser.add_argument(
        "--I-auc",
        action="store_true",
        default=False,
        help="Run AUC vs I (max number of iterations) experiments",
    )
    parser.add_argument(
        "--delta-auc",
        action="store_true",
        default=False,
        help="Run AUC vs delta (convergence algorithm error parameter) experiments",
    )
    parser.add_argument(
        "--g-auc", action="store_true", default=False, help="Run AUC vs g experiments"
    )
    parser.add_argument(
        "--stdev-I",
        action="store_true",
        default=False,
        help="Vary number of iters and measure the stdev and AUC",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save results"
    )
    parser.add_argument(
        "--params-csv",
        type=str,
        default="./spreadsheets/parameters.csv",
        help="CSV file containing kernel parameters and dataset names",
    )
    parser.add_argument(
        "--gkm-mode",
        type=str,
        choices=["dna", "protein"],
        default="dna",
        help="Whether gkm is currently compiled for protein or dna",
    )

    return parser.parse_args()


args = get_args()
df = pd.read_csv(args.params_csv)
params = df.to_dict("records")

if not osp.exists(args.output_dir):
    os.makedirs(args.output_dir)


def thread_experiment(dataset, g, m, k):
    output_csv = dataset + "_vary_threads_I50.csv"
    results = {
        "fastsk_exact_time": [],
        "fastsk_approx_time": [],
        "fastsk_approx_time_t1": [],
        "fastsk_I50": [],
        "gkm_time": [],
    }

    train_file = osp.join("./data", dataset + ".train.fasta")
    reader = FastaUtility()
    Xtrain, Ytrain = reader.read_data(train_file)

    for t in range(1, 21):
        # fastsk_exact = time_fastsk(g, m, t, FASTSK_DATA, prefix=dataset, approx=False)
        # fastsk_approx = time_fastsk(g, m, t, FASTSK_DATA, prefix=dataset, approx=True)
        # fastsk_approx_t1 = time_fastsk(g, m, t=1, FASTSK_DATA, prefix=dataset, approx=True)
        # fastsk_approx_t1 = time_fastsk(g, m, t=1, FASTSK_DATA, prefix=dataset, approx=True)
        # gkm = time_gkm(g, m, t, GKM_DATA, GKM_EXEC, prefix=dataset)
        fastsk_exact = 0
        fastsk_approx = 0
        fastsk_approx_t1 = 0
        fastsk_approx_t1 = 0
        gkm = 0
        fastsk_I50 = time_fastsk(
            g,
            m,
            t=1,
            data_location=FASTSK_DATA,
            prefix=dataset,
            approx=True,
            max_iters=50,
        )

        results["fastsk_exact_time"].append(fastsk_exact)
        results["fastsk_approx_time"].append(fastsk_approx)
        results["fastsk_approx_time_t1"].append(fastsk_approx_t1)
        results["fastsk_I50"].append(fastsk_I50)
        results["gkm_time"].append(gkm)

        # log_str = "{} - exact: {}, approx: {} approx_t1: {}, gkm: {}"
        # print(log_str.format(dataset, fastsk_exact, fastsk_approx, fastsk_approx_t1, gkm))
        log_str = "{} - I50: {}"
        print(log_str.format(dataset, fastsk_I50))

        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)


def run_thread_experiments(params):
    for p in params:
        dataset, type_, g, m, k = p["Dataset"], p["type"], p["g"], p["m"], p["k"]
        assert k == g - m
        if type_ == "dna":
            thread_experiment(dataset, g, m, k)


def m_time_experiment(dataset, output_dir):
    """January 2020: create timing figures by varying m and
    keeping g fixed

    Results collected:
        - FastSK-Exact 20 thread
        - FastSK-Approx 1 thread
        - FastSK-Approx 1 thread no variance 50 iters
        - gkmSVM-Exact 20 thread
        - gkmSVM-Approx 20 thread
    """
    output_csv = osp.join(output_dir, dataset + "_vary_m.csv")
    results = {
        "g": [],
        "k": [],
        "m": [],
        "FastSK-Exact 20 thread": [],
        "FastSK-Approx 1 thread": [],
        "FastSK-Approx 1 thread no variance 50 iters": [],
        "gkmSVM-Exact 20 thread": [],
        "gkmSVM-Approx 20 thread": [],
    }

    train_file = osp.join("./data", dataset + ".train.fasta")
    reader = FastaUtility()
    Xtrain, Ytrain = reader.read_data(train_file)

    g = 16
    min_m, max_m = 0, g - 2

    # flags to skip results that have timed out
    skip_fastsk_exact = False
    skip_fastsk_approx = False
    skip_fastsk_approx_no_var = False
    skip_gkm_exact = False
    skip_gkm_approx = False

    for m in range(min_m, max_m + 1):
        k = g - m
        max_I = int(special.comb(g, m))

        fastsk_exact, fastsk_approx, fastsk_approx_no_var = [0] * 3
        gkm_exact, gkm_approx = [0] * 2

        ## FastSK-Exact
        if not skip_fastsk_exact:
            fastsk_exact = time_fastsk(
                g,
                m,
                t=20,
                data_location=FASTSK_DATA,
                prefix=dataset,
                approx=False,
                timeout=TIMEOUT,
            )
            if fastsk_exact >= MAXTIME and m > 4:
                skip_fastsk_exact = True

        ## FastSK-Approx, iterate until convergence is reached
        if not skip_fastsk_approx:
            fastsk_approx = time_fastsk(
                g,
                m,
                t=1,
                data_location=FASTSK_DATA,
                prefix=dataset,
                approx=True,
                max_iters=max_I,
                timeout=TIMEOUT,
            )
            if fastsk_approx >= MAXTIME and m > 4:
                skip_fastsk_approx = True

        ## FastSK-Approx, iterate up to 50 iters, don't compute variance
        if not skip_fastsk_approx_no_var:
            fastsk_approx_no_var = time_fastsk(
                g,
                m,
                t=1,
                data_location=FASTSK_DATA,
                prefix=dataset,
                approx=True,
                max_iters=50,
                skip_variance=True,
                timeout=TIMEOUT,
            )
            if fastsk_approx_no_var >= MAXTIME and m > 4:
                skip_fastsk_approx_no_var = True

        ## gkm-Exact
        if not skip_gkm_exact:
            gkm_exact = time_gkm(
                g,
                m,
                t=20,
                gkm_data=GKM_DATA,
                gkm_exec=GKM_EXEC,
                prefix=dataset,
                approx=False,
                alphabet=GKM_PROT_DICT,
                timeout=TIMEOUT,
            )
            if gkm_exact >= MAXTIME and m > 4:
                skip_gkm_exact = True

        ## gkm-Approx, max_d = 3
        if not skip_gkm_approx:
            gkm_approx = time_gkm(
                g,
                m,
                t=20,
                gkm_data=GKM_DATA,
                gkm_exec=GKM_EXEC,
                prefix=dataset,
                approx=True,
                alphabet=GKM_PROT_DICT,
                timeout=TIMEOUT,
            )
            if gkm_approx >= MAXTIME and m > 4:
                skip_gkm_approx = True

        ## Save results
        results["g"].append(g)
        results["k"].append(k)
        results["m"].append(m)
        results["FastSK-Exact 20 thread"].append(fastsk_exact)
        results["FastSK-Approx 1 thread"].append(fastsk_approx)
        results["FastSK-Approx 1 thread no variance 50 iters"].append(
            fastsk_approx_no_var
        )
        results["gkmSVM-Exact 20 thread"].append(gkm_exact)
        results["gkmSVM-Approx 20 thread"].append(gkm_approx)

        print("{}: g = {}, m = {}".format(dataset, g, m))

        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)


def run_m_time_experiments(params, output_dir):
    """January 2020: create timing figures by varying m and
    keeping g fixed.
    Note: need to run DNA and protein experiments separately,
    since gkm needs to be recompiled to handle protein data
    """
    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    for p in params:
        dataset, type_ = p["Dataset"], p["type"]
        if type_ == "protein":
            m_time_experiment(dataset, output_dir)


def g_time_experiment(dataset):
    """Dec 14 experiments:
    - FastSK-Exact 20 thread
    - FastSK-Approx Approx 1 thread (convergence)
    - FastSK-Approx 20 thread (no convergence, 50 iters)
    - gkm-Exact 20 thread
    - gkm-Approx 20 thread
    - fastsk (max threads)
    """
    output_csv = osp.join(output_dir, dataset + "_g_times_dec14.csv")
    results = {
        "g": [],
        "k": [],
        "m": [],
        "FastSK-Exact": [],
        "FastSK-Approx 1 thread": [],
        "FastSK-Approx 20 thread no variance 50 iters": [],
        "gkm-Approx 20 thread": [],
        "fastsk": [],
    }

    train_file = osp.join("./data", dataset + ".train.fasta")
    reader = FastaUtility()
    Xtrain, Ytrain = reader.read_data(train_file)

    min_g, max_g = 6, 20
    k = 6

    skip_fastsk_exact = False
    skip_fastsk_approx_t1 = False
    skip_fastsk_approx_t20_no_var = False

    skip_gkm_exact = False
    skip_gkm_approx_t20 = False
    skip_fastsk = False

    for g in range(min_g, max_g + 1):
        m = g - k
        max_I = int(special.comb(g, m))

        fastsk_exact = 0
        fastsk_approx_t1 = 0
        fastsk_approx_t20_no_var = 0
        gkm_exact = 0
        gkm_approx_t20 = 0
        fastsk = 0

        ## FastSK-Exact
        if not skip_fastsk_exact:
            fastsk_exact = time_fastsk(
                g,
                m,
                t=20,
                data_location=FASTSK_DATA,
                prefix=dataset,
                approx=False,
                timeout=TIMEOUT,
            )
            if fastsk_exact >= MAXTIME and g > 8:
                skip_fastsk_exact = True

        ## FastSK-Approx, iterate until convergence is reached
        if not skip_fastsk_approx_t1:
            fastsk_approx_t1 = time_fastsk(
                g,
                m,
                t=1,
                data_location=FASTSK_DATA,
                prefix=dataset,
                approx=True,
                max_iters=max_I,
                timeout=TIMEOUT,
            )
            if fastsk_approx_t1 >= MAXTIME and g > 8:
                skip_fastsk_approx_t1 = True

        ## FastSK-Approx, don't make convergence calculations; iterate 50 times
        if not skip_fastsk_approx_t20_no_var:
            fastsk_approx_t20_no_var = time_fastsk(
                g,
                m,
                t=20,
                data_location=FASTSK_DATA,
                prefix=dataset,
                approx=True,
                max_iters=50,
                skip_variance=True,
                timeout=TIMEOUT,
            )
            if fastsk_approx_t20_no_var >= MAXTIME and g > 8:
                skip_fastsk_approx_t20_no_var = True

        ## gkm-Exact
        if not skip_gkm_exact:
            gkm_exact = time_gkm(
                g,
                m,
                t=20,
                gkm_data=GKM_DATA,
                gkm_exec=GKM_EXEC,
                prefix=dataset,
                approx=False,
                timeout=TIMEOUT,
                alphabet=PROT_DICT,
            )
            if gkm_exact >= MAXTIME and g > 8:
                skip_gkm_exact = True

        ## gkm-Approx, max_d = 3
        if not skip_gkm_approx_t20:
            gkm_approx_t20 = time_gkm(
                g,
                m,
                t=20,
                gkm_data=GKM_DATA,
                gkm_exec=GKM_EXEC,
                prefix=dataset,
                approx=True,
                timeout=TIMEOUT,
                alphabet=PROT_DICT,
            )
            if gkm_approx_t20 >= MAXTIME and g > 8:
                skip_gkm_approx_t20 = True

        if not skip_fastsk and m > 0:
            fastsk = time_fastsk(g, m, type_=type_, prefix=dataset, timeout=None)
            if fastsk >= MAXTIME and g > 8:
                skip_fastsk = True

        ## Save results
        results["g"].append(g)
        results["k"].append(k)
        results["m"].append(m)
        results["FastSK-Exact"] = fastsk_exact
        results["FastSK-Approx 1 thread"].append(fastsk_approx_t1)
        results["FastSK-Approx 20 thread no variance 50 iters"].append(
            fastsk_approx_t20_no_var
        )
        results["gkm-Approx 20 thread"].append(gkm_approx_t20)
        results["fastsk"].append(fastsk)

        print("{} - g = {}, m = {}".format(dataset, g, m))

        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)


def run_g_time_experiments(params, output_dir):
    for p in params:
        dataset, type_ = p["Dataset"], p["type"]
        if type_ == "protein":
            g_time_experiment(dataset, output_dir, type_)

def g_time_experiment_full(dataset, output_dir):
    """
    Time training and predicting for
    - FastSK-Exact 8 thread
    - FastSK-Approx 8 thread
    - gkm-Exact 8 thread
    - gkm-Approx 8 thread
    - LSGKM-Exact 8 thread
    - LSGKM-Approx 8 thread
    """
    output_csv = osp.join(output_dir, dataset + "_g_times_dec14.csv")
    results = {
        "g": [],
        "k": [],
        "m": [],
        "FastSK-Exact": [],
        "FastSK-Approx": [],
        "gkm-Exact": [],
        "gkm-Approx": [],
        "LSGKM-Exact": [], 
        "LSGKM-Approx": [],
    }

    # min_g, max_g = 6, 20
    # k = 
    
    min_g, max_g = 11, 11
    k = 7

    skip_fastsk_exact = False
    skip_fastsk_approx = False

    skip_gkm_exact = True
    skip_gkm_approx = True 

    skip_lsgkm_exact = False
    skip_lsgkm_approx = True

    for g in range(min_g, max_g + 1):
        m = g - k
        max_I = int(special.comb(g, m))

        fastsk_exact_acc, fastsk_exact_auc, fastsk_exact = 0, 0, 0
        fastsk_approx_acc, fastsk_approx_auc, fastsk_approx = 0, 0, 0
        gkm_exact_acc, gkm_exact_auc, gkm_exact = 0, 0, 0
        gkm_approx_acc, gkm_approx_auc, gkm_approx = 0, 0, 0
        lsgkm_exact_acc, lsgkm_exact_auc, lsgkm_exact = 0, 0, 0
        lsgkm_approx_acc, lsgkm_approx_auc, lsgkm_approx = 0, 0, 0

        ## FastSK-Exact
        if not skip_fastsk_exact:
            fastsk_exact_acc, fastsk_exact_auc, fastsk_exact = train_and_test_fastsk(
                dataset=dataset,
                g=g,
                m=m,
                t=4,
                approx=False,
                timeout=TIMEOUT,
            )
            if fastsk_exact >= MAXTIME and g > 8:
                skip_fastsk_exact = True
            print(f"Accuracy = {fastsk_exact_acc}, AUC = {fastsk_exact_auc}")

        ## FastSK-Approx
        if not skip_fastsk_approx:
            fastsk_approx_acc, fastsk_approx_auc, fastsk_approx = train_and_test_fastsk(
                dataset=dataset,
                g=g,
                m=m,
                t=4,
                approx=True,
                timeout=TIMEOUT,
            )
            if fastsk_approx >= MAXTIME and g > 8:
                skip_fastsk_approx = True
            print(f"Accuracy = {fastsk_approx_acc}, AUC = {fastsk_approx_auc}")

        ## gkm-Exact
        if not skip_gkm_exact:
            gkm_exact_acc, gkm_exact_auc, gkm_exact = train_and_test_gkm(
                g,
                m,
                t=4,
                gkm_data=GKM_DATA,
                gkm_exec=GKM_EXEC,
                prefix=dataset,
                approx=False,
                timeout=TIMEOUT,
                alphabet=PROT_DICT,
            )
            if gkm_exact >= MAXTIME and g > 8:
                skip_gkm_exact = True

        ## gkm-Approx, max_d = 3
        if not skip_gkm_approx:
            gkm_approx_acc, gkm_approx_auc, gkm_approx = train_and_test_gkm(
                g,
                m,
                t=4,
                gkm_data=GKM_DATA,
                gkm_exec=GKM_EXEC,
                prefix=dataset,
                approx=True,
                timeout=TIMEOUT,
                alphabet=PROT_DICT,
            )
            if gkm_approx >= MAXTIME and g > 8:
                skip_gkm_approx = True

        ## LSGKM-Exact
        if not skip_lsgkm_exact:
            lsgkm_exact_acc, lsgkm_exact_auc, lsgkm_exact = train_and_test_lsgkm(
                g,
                m,
                t=4,
                lsgkm_data=LSGKM_DATA,
                lsgkm_exec=LSGKM_EXEC,
                prefix=dataset,
                approx=False,
                timeout=TIMEOUT,
                alphabet=PROT_DICT,
            )
            if lsgkm_exact >= MAXTIME and g > 8:
                skip_lsgkm_exact = True

        ## LSGKM-Approx
        if not skip_lsgkm_approx:
            lsgkm_approx_acc, lsgkm_approx_auc, lsgkm_approx = train_and_test_lsgkm(
                g,
                m,
                t=4,
                lsgkm_data=LSGKM_DATA,
                lsgkm_exec=LSGKM_EXEC,
                prefix=dataset,
                approx=True,
                timeout=TIMEOUT,
                alphabet=PROT_DICT,
            )
            if lsgkm_approx >= MAXTIME and g > 8:
                skip_lsgkm_approx = True
    

        ## Save results
        results["g"].append(g)
        results["k"].append(k)
        results["m"].append(m)
        results["FastSK-Exact"].append((fastsk_exact_acc, fastsk_exact_auc, fastsk_exact))
        results["FastSK-Approx"].append((fastsk_approx_acc, fastsk_approx_auc, fastsk_approx))
        results["gkm-Exact"].append((gkm_exact_acc, gkm_exact_auc, gkm_exact))
        results["gkm-Approx"].append((gkm_approx_acc, gkm_approx_auc, gkm_approx))
        results["LSGKM-Exact"].append((lsgkm_exact_acc, lsgkm_exact_auc, lsgkm_exact))
        results["LSGKM-Approx"].append((lsgkm_approx_acc, lsgkm_approx_auc, lsgkm_approx ))

        print("{} - g = {}, m = {}".format(dataset, g, m))
        print(results)

        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)


def run_g_time_experiments_full(params, output_dir):
    for p in params:
        dataset, type_ = p["Dataset"], p["type"]
        if type_ == "dna":
            g_time_experiment_full(dataset, output_dir)
            break


def I_experiment(dataset, g, m, k, C):
    output_csv = dataset + "_vary_I.csv"
    results = {
        "I": [],
        "acc": [],
        "auc": [],
    }

    if m == 0:
        m = 1

    max_I = min(int(special.comb(g, m)), 100)
    iter_vals = []
    if max_I > 10:
        iter_vals += list(range(1, 10))
        iter_vals += list(range(10, max_I, 10))
        iter_vals.append(max_I)
    else:
        iter_vals = list(range(1, max_I + 1))

    for I in iter_vals:
        fastsk = FastskRunner(dataset)
        acc, auc, _ = fastsk.train_and_test(
            g, m, t=1, approx=True, I=I, delta=0.025, C=C
        )
        log_str = "{}: I = {}, auc = {}, acc = {}".format(dataset, I, auc, acc)
        print(log_str)
        results["I"].append(I)
        results["acc"].append(acc)
        results["auc"].append(auc)

        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)


def run_I_experiments(params):
    for p in params:
        dataset, type_, g, m, k, C = (
            p["Dataset"],
            p["type"],
            p["g"],
            p["m"],
            p["k"],
            p["C"],
        )
        # if dataset in ['ZZZ3', 'KAT2B', 'EP300_47848']:
        #     continue
        assert k == g - m
        I_experiment(dataset, g, m, k, C)


def delta_experiment(dataset, g, m, k, C):
    output_csv = dataset + "_vary_delta.csv"
    results = {
        "delta": [],
        "acc": [],
        "auc": [],
    }

    max_I = int(special.comb(g, m))
    delta_vals = [0.005 * i for i in range(20)] + [0.1 * i for i in range(1, 11)]
    for d in delta_vals:
        fastsk = FastskRunner(dataset)
        acc, auc, _ = fastsk.train_and_test(
            g, m, t=1, approx=True, I=max_I, delta=d, C=C
        )
        log_str = "{}: d = {}, acc = {}, auc = {}".format(dataset, d, acc, auc)
        print(log_str)
        results["delta"].append(d)
        results["acc"].append(acc)
        results["auc"].append(auc)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)


def run_delta_experiments(params):
    for p in params:
        dataset, type_, g, m, k, C = (
            p["Dataset"],
            p["type"],
            p["g"],
            p["m"],
            p["k"],
            p["C"],
        )
        assert k == g - m
        if dataset in ["ZZZ3", "KAT2B", "EP300_47848"]:
            continue
        delta_experiment(dataset, g, m, k, C)


def check_C_vals(g, m, dataset):
    best_auc, best_acc, best_C = 0, 0, 0
    C_vals = [10 ** i for i in range(-3, 3)]
    max_I = max_I = min(int(special.comb(g, m)), 100)
    for C in C_vals:
        fastsk = FastskRunner(dataset)
        acc, auc, _ = fastsk.train_and_test(g, m, t=1, I=max_I, approx=True, C=C)
        if auc > best_auc:
            best_acc, best_auc = acc, auc
    return best_acc, best_auc, C


def g_auc_experiment(dataset, output_dir, C, type_):
    print("Running g_auc_experiments on {}".format(dataset))
    assert type_ in ["dna", "protein"]
    output_csv = osp.join(output_dir, dataset + "_dec15_g_auc.csv")
    results = {
        "g": [],
        "k": [],
        "m": [],
        "fastsk_approx_conv_acc": [],
        "fastsk_approx_conv_auc": [],
        #'fastsk_approx_i50_acc': [],
        #'fastsk_approx_i50_auc': [],
        #'gkm_approx_acc': [],
        #'gkm_approx_auc': [],
        "lsgkm_approx_acc": [],
        "lsgkm_approx_auc": []
    }

    train_file = osp.join("../data", dataset + ".train.fasta")
    test_file = osp.join("../data", dataset + ".test.fasta")

    fasta_util = FastaUtility()
    max_g = min(
        fasta_util.shortest_seq(train_file), fasta_util.shortest_seq(test_file), 20
    )
    k = 6

    gkm_alphabet = GKM_PROT_DICT if type_ == "protein" else None

    skip_fastsk, skip_gkm, skip_lsgkm = False, True, False

    for g in range(k, max_g + 1):
        #### Run experiments
        m = g - k

        ## FastSK-Approx with up to 50 iterations/mismatch combos
        fastsk = FastskRunner(dataset)
        if not skip_fastsk:
            fsk_acc, fsk_auc, fsk_time = train_and_test_fastsk(
                dataset,
                g,
                m,
                t=1,
                I=100,
                approx=True,
                skip_variance=False,
                C=C,
                timeout=TIMEOUT,
            )
            if fsk_time >= TIMEOUT:
                skip_fastsk = True
        else:
            fsk_acc, fsk_auc = 0, 0

        ## gkm-Approx (m_max = 3)
        if not skip_gkm:
            gkm_approx_acc, gkm_approx_auc, gkmtime = train_and_test_gkm(
                g=g,
                m=m,
                t=20,
                prefix=dataset,
                gkm_data=GKM_DATA,
                gkm_exec=GKM_EXEC,
                approx=True,
                timeout=TIMEOUT,
                alphabet=gkm_alphabet,
            )
            if gkmtime >= TIMEOUT:
                skip_gkm = True
        else:
            gkm_approx_acc, gkm_approx_auc = 0, 0
            
        if not skip_lsgkm:
            lsgkm_approx_acc, lsgkm_approx_auc, lsgkmtime = train_and_test_lsgkm(
                g=g,
                m=m,
                t=20,
                prefix=dataset,
                lsgkm_data=LSGKM_DATA,
                lsgkm_exec=LSGKM_EXEC,
                approx=True,
                timeout=TIMEOUT,
                alphabet=gkm_alphabet,
            )
            if lsgkmtime >= TIMEOUT:
                skip_lsgkm = True
        else:
            lsgkm_approx_acc, lsgkm_approx_auc = 0, 0

        #### Log results

        log_str = "\n\ng = {}, m = {}, fastsk auc = {}, gkm approx auc = {}, lsgkm approx auc = {}\n\n"
        print(log_str.format(g, m, fsk_auc, gkm_approx_auc, lsgkm_approx_auc))

        results["g"].append(g)
        results["k"].append(k)
        results["m"].append(m)
        results["fastsk_approx_conv_acc"].append(fsk_acc)
        results["fastsk_approx_conv_auc"].append(fsk_auc)
        # results['gkm_approx_acc'].append(gkm_approx_acc)
        # results['gkm_approx_auc'].append(gkm_approx_auc)
        results['lsgkm_approx_acc'].append(lsgkm_approx_acc)
        results['lsgkm_approx_auc'].append(lsgkm_approx_auc)

        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)


def run_g_auc_experiments(params, output_dir):
    for p in params:
        dataset, type_, g, m, k, C = (
            p["Dataset"],
            p["type"],
            p["g"],
            p["m"],
            p["k"],
            p["C"],
        )
        assert k == g - m
        if type_ == "dna":
            g_auc_experiment(dataset, output_dir, C, type_)


def fastsk_fastsk_protein_kernel_times(params):
    output_csv = "fastsk_fastsk_protein_kernel_times.csv"
    results = {
        "dataset": [],
        "g": [],
        "m": [],
        "k": [],
        "fastsk_exact": [],
        "fastsk_approx_t1": [],
        "fastsk_I50": [],
        "fastsk": [],
    }
    count = 0
    for p in params:
        dataset, type_, g, m, k = p["Dataset"], p["type"], p["g"], p["m"], p["k"]
        if type_ != "protein":
            continue

        max_I = int(special.comb(g, m))

        fastsk_exact = time_fastsk(
            g, m, t=20, data_location=FASTSK_DATA, prefix=dataset, approx=False
        )

        fastsk_approx_t1 = time_fastsk(
            g,
            m,
            t=1,
            data_location=FASTSK_DATA,
            prefix=dataset,
            approx=True,
            max_iters=max_I,
        )

        fastsk_I50 = time_fastsk(
            g,
            m,
            t=1,
            data_location=FASTSK_DATA,
            prefix=dataset,
            approx=True,
            max_iters=50,
        )

        fastsk = time_fastsk(g, m, type_="protein", prefix=dataset)

        results["dataset"].append(dataset)
        results["g"].append(g)
        results["m"].append(m)
        results["k"].append(k)
        results["fastsk_exact"].append(fastsk_exact)
        results["fastsk_approx_t1"].append(fastsk_approx_t1)
        results["fastsk_I50"].append(fastsk_I50)
        results["fastsk"].append(fastsk)

        for key in results:
            print("{} - {}".format(key, results[key][count]))

        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)

        count += 1


def fastsk_gkm_dna_kernel_times(params):
    output_csv = "fastsk_gkm_dna_kernel_times_part2.csv"
    results = {
        "dataset": [],
        "g": [],
        "m": [],
        "k": [],
        "fastsk_exact": [],
        "fastsk_approx_t1": [],
        "fastsk_I50": [],
        "gkm_exact": [],
        "gkm_approx": [],
    }
    count = 0
    for p in params:
        dataset, type_, g, m, k = p["Dataset"], p["type"], p["g"], p["m"], p["k"]
        if type_ != "dna":
            continue

        if dataset not in ["KAT2B", "TP53", "ZZZ3"]:
            continue

        max_I = int(special.comb(g, m))

        fastsk_exact = time_fastsk(
            g, m, t=20, data_location=FASTSK_DATA, prefix=dataset, approx=False
        )

        fastsk_approx_t1 = time_fastsk(
            g,
            m,
            t=1,
            data_location=FASTSK_DATA,
            prefix=dataset,
            approx=True,
            max_iters=max_I,
        )

        fastsk_I50 = time_fastsk(
            g,
            m,
            t=1,
            data_location=FASTSK_DATA,
            prefix=dataset,
            approx=True,
            max_iters=50,
        )

        gkm_exact = time_gkm(
            g,
            m,
            t=20,
            gkm_data=GKM_DATA,
            gkm_exec=GKM_EXEC,
            prefix=dataset,
            approx=False,
        )

        gkm_approx = time_gkm(
            g, m, t=1, gkm_data=GKM_DATA, gkm_exec=GKM_EXEC, prefix=dataset, approx=True
        )

        results["dataset"].append(dataset)
        results["g"].append(g)
        results["m"].append(m)
        results["k"].append(k)
        results["fastsk_exact"].append(fastsk_exact)
        results["fastsk_approx_t1"].append(fastsk_approx_t1)
        results["fastsk_I50"].append(fastsk_I50)
        results["gkm_exact"].append(gkm_exact)
        results["gkm_approx"].append(gkm_approx)

        for key in results:
            print("{} - {}".format(key, results[key][count]))

        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)

        count += 1


def fastsk_blended_nlp_kernel_times(params):
    output_csv = "fastsk_blended_nlp_kernel_times.csv"
    results = {
        "dataset": [],
        "g": [],
        "m": [],
        "k": [],
        "k1": [],
        "k2": [],
        "fastsk_exact": [],
        "fastsk_approx_t1": [],
        "fastsk_I50": [],
        "blended": [],
    }
    count = 0
    k1, k2 = 3, 5
    for p in params:
        dataset, type_, g, m, k = p["Dataset"], p["type"], p["g"], p["m"], p["k"]
        if type_ != "nlp":
            continue

        max_I = int(special.comb(g, m))

        fastsk_exact = time_fastsk(
            g, m, t=20, data_location=FASTSK_DATA, prefix=dataset, approx=False
        )

        fastsk_approx_t1 = time_fastsk(
            g,
            m,
            t=1,
            data_location=FASTSK_DATA,
            prefix=dataset,
            approx=True,
            max_iters=max_I,
        )

        fastsk_I50 = time_fastsk(
            g,
            m,
            t=1,
            data_location=FASTSK_DATA,
            prefix=dataset,
            approx=True,
            max_iters=50,
        )

        blended = time_blended(k1=k1, k2=k2, prefix=dataset)

        results["dataset"].append(dataset)
        results["g"].append(g)
        results["m"].append(m)
        results["k"].append(k)
        results["k2"].append(k1)
        results["k1"].append(k2)
        results["fastsk_exact"].append(fastsk_exact)
        results["fastsk_approx_t1"].append(fastsk_approx_t1)
        results["fastsk_I50"].append(fastsk_I50)
        results["blended"].append(blended)

        for key in results:
            print("{} - {}".format(key, results[key][count]))

        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)

        count += 1


def get_CI(data, confidence=0.95):
    n = len(data)
    mean_ = mean(data)
    std_err = sem(data)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    lower = mean_ - h
    upper = mean_ + h
    return mean_, lower, upper


def stdev_and_auc_vs_iters_experiments(params, output_dir):
    """Given a dictionary that provide the dataset to use and
    the parameter set to use, vary the number of iterations.
    At each number of iters, get the stdev of the approximated kernel matrix.
    Then, train and evaluate a model. Obtain the test AUC.
    """

    # Get parameters
    print(params)
    dataset, type_ = params["Dataset"], params["type"]
    g, m, k, C = params["g"], params["m"], params["k"], params["C"]
    assert k == g - m
    assert g > 0

    results = {"dataset": dataset, "g": g, "k": k, "m": m, "C": C, "iters": []}

    for i in range(5):
        results["acc sample {}".format(i + 1)] = []
    results["mean acc"] = []
    results["lower acc"] = []
    results["upper acc"] = []
    for i in range(5):
        results["auc sample {}".format(i + 1)] = []
    results["mean auc"] = []
    results["lower auc"] = []
    results["upper auc"] = []
    for i in range(5):
        results["stdev sample {}".format(i + 1)] = []
    results["mean stdev"] = []
    results["lower stdev"] = []
    results["upper stdev"] = []

    # max iters - capped at 500
    max_I = min(int(special.comb(g, m)), 500)

    iters = [1]

    if max_I > 1:
        # increment by 2 for first 10
        iters += list(range(2, min(max_I, 10), 2))
        # increment by 5 until 50
        if max_I >= 10:
            iters += list(range(10, min(max_I, 50), 5))
        # increment by 20 until 500
        if max_I >= 50:
            iters += list(range(50, max_I, 20))
        # include max
        iters += [max_I]

    for I in tqdm(iters):
        results["iters"].append(I)
        sample_accs, sample_aucs, sample_stdevs = [], [], []
        for i in range(5):
            fastsk = FastskRunner(dataset)
            acc, auc = fastsk.train_and_test(
                g, m, t=1, approx=True, I=I, delta=0.025, C=C
            )
            stdevs = fastsk.stdevs
            assert len(stdevs) == I
            stdev = stdevs[-1]

            log_str = "{}: I = {}, auc = {}, acc = {}, stdevs = {}"
            log_str = log_str.format(dataset, I, auc, acc, stdev)
            print(log_str)

            sample_accs.append(acc)
            sample_aucs.append(auc)
            sample_stdevs.append(stdev)

            results["acc sample {}".format(i + 1)].append(acc)
            results["auc sample {}".format(i + 1)].append(auc)
            results["stdev sample {}".format(i + 1)].append(stdev)

        confidence = 0.95
        n = len(sample_accs)

        mean_acc, lower_acc, upper_acc = get_CI(sample_accs, confidence=0.95)
        mean_auc, lower_auc, upper_auc = get_CI(sample_aucs, confidence=0.95)
        mean_stdev, lower_stdev, upper_stdev = get_CI(sample_stdevs, confidence=0.95)

        results["mean acc"].append(mean_acc)
        results["lower acc"].append(lower_acc)
        results["upper acc"].append(upper_acc)
        results["mean auc"].append(mean_auc)
        results["lower auc"].append(lower_auc)
        results["upper auc"].append(upper_auc)
        results["mean stdev"].append(mean_stdev)
        results["lower stdev"].append(lower_stdev)
        results["upper stdev"].append(upper_stdev)

        df = pd.DataFrame(results)
        if not osp.exists(output_dir):
            os.makedirs(output_dir)
        output_csv = osp.join(output_dir, "{}_stdev_auc_iters.csv".format(dataset))
        df.to_csv(output_csv, index=False)


def run_stdev_and_auc_vs_iters_experiments(params, output_dir):
    """Note, these requires require that fastsk.cpp be tweaked
    such that it records the stdev values; these are not normally
    saved or provided to the user.
    """
    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    for p in params:
        dataset, type_, g, m, k = p["Dataset"], p["type"], p["g"], p["m"], p["k"]
        if type_ == args.gkm_mode:
            stdev_and_auc_vs_iters_experiments(p, output_dir)


if args.threads:
    run_thread_experiments(params)
if args.m_time:
    run_m_time_experiments(params, args.output_dir)
if args.g_time:
    run_g_time_experiments(params, args.output_dir)
if args.g_time_full:
    run_g_time_experiments_full(params, args.output_dir)
if args.I_auc:
    run_I_experiments(params)
if args.stdev_I:
    run_stdev_and_auc_vs_iters_experiments(params, args.output_dir)
if args.delta_auc:
    run_delta_experiments(params)
if args.g_auc:
    run_g_auc_experiments(params, args.output_dir)
