"""fastsk_gridsearch.py
Run a grid search over g, m, and C values to find optimal
hyper-parameters for each dataset.
"""

import os
import os.path as osp
import sys
import pandas as pd
from tqdm import tqdm

from utils import FastskRunner

min_g, max_g = 4, 15
G_VALS = list(range(min_g, max_g + 1))
C_VALS = [10 ** i for i in range(-3, 3)]
GRID = []
for C in C_VALS:
    for g in G_VALS:
        for m in range(0, g - 2):
            k = g - m
            GRID.append({"C": C, "g": g, "m": m})

DATASETS_CSV = "spreadsheets/testDatasets.csv"
OUTPUT_CSV = "gridsearch_results.csv"


def run_gridsearch(dataset):
    best_auc, best_params = 0, {}

    iterator = tqdm(GRID, desc="{} grid search".format(dataset), total=len(GRID))

    for param_vals in iterator:
        C, g, m = param_vals["C"], param_vals["g"], param_vals["m"]
        k = g - m

        fastsk = FastskRunner(dataset)
        acc, auc = fastsk.train_and_test(g, m, t=1, approx=True, C=C)

        params = {
            "dataset": dataset,
            "g": g,
            "m": m,
            "k": k,
            "C": C,
            "delta": 0.025,
            "acc": acc,
            "auc": auc,
        }

        if auc > best_auc:
            best_auc = auc
            best_params = params

    print(best_params)
    return best_params


def run_gridsearches(datasets):
    df = pd.DataFrame(
        columns=[
            "dataset",
            "g",
            "m",
            "k",
            "C",
            "delta",
            "acc",
            "auc",
        ]
    )

    iterator = tqdm(datasets, desc="Grid Searches", total=len(datasets))

    for dataset in iterator:
        best_params = run_gridsearch(dataset)
        df = df.append(best_params, ignore_index=True)
        df.to_csv(OUTPUT_CSV, index=False)


datasets = pd.read_csv(DATASETS_CSV)["dataset"].tolist()
run_gridsearches(datasets)
