'''
Derrick Blakely - September 2019

This script covers experiments on our
DNA datasets.

Note that gkm-SVM2.0 must be recompiled
with a larger alphabet size if running
on a dataset besides DNA.
'''

import subprocess

experiments = [
    {"dataset": "CTCF", "g": 13, "m": 7, "k": 6},
    {"dataset": "EP300", "g": 10, "m": 4, "k": 6},
    {"dataset": "JUND", "g": 10, "m": 3, "k": 7},
    {"dataset": "RAD21", "g": 14, "m": 8, "k": 6},
    {"dataset": "SIN3A", "g": 8, "m": 2, "k": 6},
    {"dataset": "Pbde", "g": 5, "m": 1, "k": 4},
    {"dataset": "Hek29", "g": 5, "m": 0, "k": 5},
    {"dataset": "Mcf7", "g": 5, "m": 0, "k": 5},
    {"dataset": "EP300_47848", "g": 11, "m": 5, "k": 6},
    {"dataset": "KAT2B", "g": 13, "m": 7, "k": 6},
    {"dataset": "NR2C2", "g": 10, "m": 4, "k": 6},
    {"dataset": "TP53", "g": 7, "m": 2, "k": 5},
    {"dataset": "ZBTB33", "g": 7, "m": 1, "k": 6},
    {"dataset": "ZZZ3", "g": 10, "m": 4, "k": 6},
]

for e in experiments:
    command = ["python", "run_gkm.py", 
        "--dir", "our_data/", 
        "--prefix", e["dataset"],
        "--outdir", "temp",
        "-g", str(e["g"]),
        "-m", str(e["m"]),
        "--results", "dna_results.out"]
    print(' '.join(command))
    output = subprocess.check_output(command)
