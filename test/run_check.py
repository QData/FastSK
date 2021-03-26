"""Quick validation script to compute a gapped kmer kernel
and train a classifier.
"""

import argparse

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
import numpy as np

from fastsk import FastSK
from utils import (
    FastaUtility,

)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train',
        type=str,
        default='../data/EP300.train.fasta',
        help='training sequences file'
    )
    parser.add_argument(
        '--test',
        type=str,
        default='../data/EP300.test.fasta',
        help='test sequences file'
    )
    args = parser.parse_args()

    return args

def main(args):
    ## Compute kernel matrix
    fastsk = FastSK(g=10, m=6, t=1, approx=True)
    fastsk.compute_kernel(args.train, args.test)

    Xtrain = fastsk.get_train_kernel()
    Xtest = fastsk.get_test_kernel()

    reader = FastaUtility()
    Xseq, Ytrain = reader.read_data(args.train)

    ## Use linear SVM
    svm = LinearSVC(C=1)
    clf = CalibratedClassifierCV(svm, cv=5).fit(Xtrain, Ytrain)

    ## Evaluate
    reader = FastaUtility()
    Xseq, Ytest = reader.read_data(args.test)

    acc = clf.score(Xtest, Ytest)
    probs = clf.predict_proba(Xtest)[:,1]
    auc = roc_auc_score(Ytest, probs)

    print("Linear SVM:\n\tAcc = {}, AUC = {}".format(acc, auc))
    assert auc >= 0.9, (
        "AUC is not correct. Should be >= 0.9. Received: {}".format(auc)
    )

if __name__ == '__main__':
    args = get_args()
    main(args)
