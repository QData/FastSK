'''Demo of how to use fastsk's kernel function
as an empirical kernel map (EKM) in conjunction
with scikit-learn's linear SVM classifier. 
'''

import numpy as np
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

def evaluate_clf(clf, Xtest, Ytest):
    acc = clf.score(Xtest, Ytest)
    probs = clf.predict_proba(Xtest)[:,1]
    auc = metrics.roc_auc_score(Ytest, probs)
    return acc, auc

### Read the data
reader = FastaUtility()
Xtrain, Ytrain = reader.read_data('./data/2.19.train.fasta')
Xtest, Ytest = reader.read_data('./data/2.19.test.fasta')
Ytest = np.array(Ytest).reshape(-1, 1)

### Compute the fastsk kernel
kernel = Kernel(g=6, m=2)
kernel.compute(Xtrain, Xtest)
Xtrain = kernel.train_kernel()
Xtest = kernel.test_kernel()

### Use linear SVM
svm = LinearSVC()
clf = CalibratedClassifierCV(svm, cv=5).fit(Xtrain, Ytrain)
acc, auc = evaluate_clf(clf, Xtest, Ytest)
print("Linear SVM:\n\tAcc = {}, AUC = {}".format(acc, auc))

### Use logistic regression
clf = LogisticRegression().fit(Xtrain, Ytrain)
acc, auc = evaluate_clf(clf, Xtest, Ytest)
print("Logistic Regression:\n\tAcc = {}, AUC = {}".format(acc, auc))
