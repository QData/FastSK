"""fastsk_gridsearch_for_regression.py

Run a grid search over g and m, with alpha values of Lasso regression
automatically tuned to find optimal hyper-parameters for each dataset.
"""

import os
import os.path as osp
import sys
import pandas as pd
from tqdm import tqdm

from utils import FastskRegressor

min_g, max_g = 4, 15
G_VALS = list(range(min_g, max_g + 1))
GRID = []
for g in G_VALS:
    for m in range(0, g - 2):
        k = g - m
        GRID.append({"g": g, "m": m})

DATASETS_CSV = "spreadsheets/datasets_to_use.csv"
OUTPUT_CSV = "gridsearch_results.csv"
OUTPUT_CSV_FULL = "gridsearch_results_full.csv"


def run_gridsearch(dataset):
    best_r2, best_params = 0, {}

    df = pd.DataFrame(
        columns=[
            "dataset",
            "g",
            "m",
            "k",
            "delta",
            "r2",
        ]
    )

    iterator = tqdm(GRID, desc="{} grid search".format(dataset), total=len(GRID))

    for param_vals in iterator:
        g, m = param_vals["g"], param_vals["m"]
        k = g - m

        fastsk = FastskRegressor(dataset)
        r2 = fastsk.train_and_test(g, m, t=1, approx=True)

        params = {
            "dataset": dataset,
            "g": g,
            "m": m,
            "k": k,
            "delta": 0.025,
            "r2": r2,
        }

        print(params)

        if r2 > best_r2:
            best_r2 = r2
            best_params = params

        df = df.append(params, ignore_index=True)
        df.to_csv(OUTPUT_CSV_FULL, index=False)

    print(best_params)
    return best_params


def run_gridsearches(datasets):
    df = pd.DataFrame(
        columns=[
            "dataset",
            "g",
            "m",
            "k",
            "delta",
            "r2",
        ]
    )

    iterator = tqdm(datasets, desc="Grid Searches", total=len(datasets))

    for dataset in iterator:
        best_params = run_gridsearch(dataset)
        df = df.append(best_params, ignore_index=True)
        df.to_csv(OUTPUT_CSV, index=False)


datasets = pd.read_csv(DATASETS_CSV)["dataset"].tolist()
run_gridsearches(datasets)
