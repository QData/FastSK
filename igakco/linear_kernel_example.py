'''Demo of how to use igakco's kernel function
as an empirical kernel map (EKM) in conjunction
with scikit-learn's linear SVM classifier. 
'''

from igakco import Kernel
from sklearn.svm import LinearSVC
from utils import FastaUtility

'''For a linear kernel, we need the training and
testing data ahead of time. With igakco, we can use
an empirical kernel map (EKM) as the features to a linear
classify.
'''
reader = FastaUtility()
Xtrain, Ytrain = reader.read_data('./small.train.fasta')
Xtest, Ytest = reader.read_data('./small.test.fasta')

# Compute the igakco kernel
kernel = Kernel(g=3, m=1)
kernel.compute(Xtrain, Xtest)
Xtrain = kernel.train_kernel()
Xtest = kernel.test_kernel()
