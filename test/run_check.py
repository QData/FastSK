from fastsk import FastSK
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
from utils import *

import argparse

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--train', default='../data/EP300.train.fasta',
		help='training sequences file')
	parser.add_argument('--test', default='../data/EP300.test.fasta',
		help='test sequences file')

	args = parser.parse_args()

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
	auc = metrics.roc_auc_score(Ytest, probs)

	print("Linear SVM:\n\tAcc = {}, AUC = {}".format(acc, auc))