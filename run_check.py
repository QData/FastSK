from fastsk import FastSK
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

if __name__ == '__main__':
	## Compute kernel matrix
	fastsk = FastSK(g=10, m=6, t=1, approx=True)
	fastsk.compute_kernel('data/EP300.train.fasta', 'data/EP300.test.fasta')

	Xtrain = fastsk.get_train_kernel()
	Xtest = fastsk.get_test_kernel()

	## Use linear SVM
	svm = LinearSVC(C=1)
	clf = CalibratedClassifierCV(svm, cv=5).fit(Xtrain, Ytrain)

	## Evaluate
	acc = clf.score(Xtest, Ytest)
	probs = clf.predict_proba(Xtest)[:,1]
	auc = metrics.roc_auc_score(Ytest, probs)

	print("Linear SVM:\n\tAcc = {}, AUC = {}".format(acc, auc))