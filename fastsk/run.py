'''Demo of how to use fastsk's kernel function
as an empirical kernel map (EKM) in conjunction
with scikit-learn's linear SVM classifier. 
'''
import numpy as np
import argparse
from fastsk import Kernel
from utils import FastaUtility
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn import metrics

'''For a linear kernel, we need the training and
testing data ahead of time. With fastsk, we can use
an empirical kernel map (EKM) as the features to a linear
classify.
'''

def get_args():
    parser = argparse.ArgumentParser(description='fastsk Evaluations')
    parser.add_argument('--trn', type=str, 
        required=True, help='Training file', metavar='1.1.train.fasta')
    parser.add_argument('--tst', type=str, 
        required=True, help='Test file', metavar='1.1.test.fasta')
    parser.add_argument('-g', type=int, required=True)
    parser.add_argument('-m', type=int, required=True)
    parser.add_argument('-C', type=float, required=False, default=1)
    parser.add_argument('-t', type=int, required=False, default=20,
        help="Numbner of threads to use for kernel computation")
    parser.add_argument('-a', '--approx', action='store_true', default=False,
        help="Flag to enable the approximation algorithm")
    parser.add_argument('-I', type=int, required=False, default=50,
        help='Maximum number of iterations to use if running the approximation algorithm')

    return parser.parse_args()

def evaluate_clf(clf, Xtest, Ytest):
    acc = clf.score(Xtest, Ytest)
    probs = clf.predict_proba(Xtest)[:,1]
    auc = metrics.roc_auc_score(Ytest, probs)
    return acc, auc

args = get_args()
train_file, test_file = args.trn, args.tst
g, m, C, t, approx, I = args.g, args.m, args.C, args.t, args.approx, args.I

### Read the data
reader = FastaUtility()
Xtrain, Ytrain = reader.read_data(train_file)
Xtest, Ytest = reader.read_data(test_file)
Ytest = np.array(Ytest).reshape(-1, 1)

### Compute the fastsk kernel
kernel = Kernel(g=g, m=m, t=t, approx=approx, max_iters=I)
kernel.compute(Xtrain, Xtest)
Xtrain = kernel.train_kernel()
Xtest = kernel.test_kernel()

### Use linear SVM
svm = LinearSVC(C=C)
clf = CalibratedClassifierCV(svm, cv=5).fit(Xtrain, Ytrain)
acc, auc = evaluate_clf(clf, Xtest, Ytest)
print("Linear SVM:\n\tAcc = {}, AUC = {}".format(acc, auc))
