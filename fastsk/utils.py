import os
import os.path as osp
import subprocess
import numpy as np
from fastsk import Kernel
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn import metrics
import time
import multiprocessing
import subprocess

def time_fastsk(g, m, t, data_location, prefix, approx=False, max_iters=None, timeout=None, skip_variance=False):
    '''Run FastGSK kernel computation. If a timeout is provided,
    it'll run as a subprocess, which will be killed when the timeout is
    reached.
    '''
    fastsk = FastskRunner(prefix, data_location)
    
    start = time.time()
    if timeout:
        if max_iters:
            args = {'t': t, 'approx': approx, 'skip_variance': skip_variance, 'I': max_iters}
        else:
            args = {'t': t, 'approx': approx, 'skip_variance': skip_variance}
        p = multiprocessing.Process(target=fastsk.compute_train_kernel, 
            name='TimeFastSK', 
            args=(g, m),
            kwargs=args)
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            p.join()
    else:
        if max_iters:
            fastsk.compute_train_kernel(g, m, t=t, approx=approx, I=max_iters, skip_variance=skip_variance)
        else:
            fastsk.compute_train_kernel(g, m, t=t, approx=approx, skip_variance=skip_variance)

    end = time.time()
    
    return end - start

def time_gkm(g, m, t, prefix, gkm_data, gkm_exec, approx=False, timeout=None, alphabet=None):
    '''Run gkm-SVM2.0 kernel computation. If a timeout is provided,
    it'll be run as a subprocess, which will be killed when the timeout is 
    reached.
    '''
    gkm = GkmRunner(gkm_exec, gkm_data, prefix, './temp')

    start = time.time()
    if timeout:
        kwargs = {'approx': approx, 'alphabet': alphabet}
        p = multiprocessing.Process(target=gkm.compute_kernel,
            name='TimeGkm',
            args=(g, m, t),
            kwargs=kwargs)
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            p.join()
    else:
        gkm.compute_kernel(g, m, t, approx=approx, alphabet=alphabet)

    end = time.time()

    return end - start

def time_gakco(g, m, type_, prefix, timeout=None):
    gakco_exec = '/localtmp/dcb7xz/FastSK/baselines/GaKCo-SVM/bin/GaKCo'
    data = './data/'
    gakco = GaKCoRunner(gakco_exec, data, type_, prefix)

    start = time.time()
    gakco.compute_kernel(g, m, mode='train')
    end = time.time()

    return end - start

def time_blended(k1, k2, prefix, timeout=None):
    blended_exec = '/localtmp/dcb7xz/FastSK/baselines/String_Kernels_Package/code/'
    data = './data/'
    blended = BlendedSpectrumRunner(blended_exec, data, prefix)
    start = time.time()
    blended.compute_kernel(k1, k2, mode='train')
    end = time.time()

    return end - start

class Vocabulary(object):
    """A class for storing the vocabulary of a 
    sequence dataset. Maps words or characters to indexes in the
    vocabulary.
    """
    def __init__(self):
        self._token2idx = {}
        self._token2idx[0] = 0
        self._size = len(self._token2idx)

    def add(self, token):
        """
        Add a token to the vocabulary.
        Args:
            token: a letter (for char-level model) or word (for word-level model)
            for which to create a mapping to an integer (the idx).
        Return: 
            the index of the word. If it's already present, return its
            index. Otherwise, add it before returning the index.
        """
        if token not in self._token2idx:
            self._token2idx[token] = self._size
            self._size += 1
        return self._token2idx.get(token)

    def size(self):
        """Return the number tokens in the vocabulary.
        """
        return self._size

    def __str__(self):
        return str(self._token2idx)

class FastaUtility():
    def __init__(self, vocab=None):
        r"""
        Initialize a helper object for parsing datasets in FASTA-like format.

        Parameters
        ----------
        vocab : 
        """
        self._vocab = Vocabulary() if vocab is None else vocab

    def read_data(self, data_file, vocab='inferred'):
        r"""Read a file with the FASTA-like format of alternating
        labels lines followed by sequences. For example:
            >1
            >AAAGAT
            >1
            >AAAAAGAT
            >0
            >AGTC
        
        Parameters
        ----------
        data_file : string
            The path to the sequences.
        vocab : string 

        Returns
        ----------
        X : list
            list of sequences where characters have been mapped to numbers.
        Y : list
            list of labels
        """
        assert vocab.lower() in ['dna', 'protein', 'inferred']
        X, Y = [], []
        with open (data_file, 'r') as f:
            label_line = True
            for line in f:
                line = line.strip().lower()
                if label_line:
                    split = line.split('>')
                    assert len(split) == 2
                    label = int(split[1])
                    assert label in [-1, 0, 1]
                    Y.append(label)
                    label_line = False
                else:
                    seq = list(line)
                    seq = [self._vocab.add(token) for token in seq]
                    X.append(seq)
                    label_line = True
            assert len(X) == len(Y)

        return X, Y

    def shortest_seq(self, data_file):
        X, Y = self.read_data(data_file)
        shortest = len(X[0])
        for x in X:
            if len(x) < shortest:
                shortest = len(x)
        return shortest

class ArabicUtility():
    def __init__(self, vocab=None):
        r"""
        Initialize a helper object for parsing datasets in the MADAR Arabic
        Dialect Identification task format.
        https://www.aclweb.org/anthology/L18-1535/

        There are 26 dialects in one of the tasks, which is 
        too many for us to handle right now. Instead, we just the 
        following 6 cities:
            RAB - Rabat
            BEI - Beirut
            DOH - Doha
            CAI - Cairo
            TUN - Tunis
            MSA - Modern Standard Arabic

        Parameters
        ----------
        vocab : a Vocabulary object
        """
        self._vocab = Vocabulary() if vocab is None else vocab
        self._classes = Vocabulary()
        self._labels_to_use = ['RAB', 'BEI', 'DOH', 'CAI', 'TUN', 'MSA']

    def read_data(self, data_file, vocab='inferred'):
        r"""Read a file with the following format:
            بالمناسبة ، اسمي هيروش إيجيما . MSA
            مش قادر نرقد كويس في الليل .    CAI

            That is, a sequence of Arabic characters, a tab,
            and a three-letter label/city code.
        
        Parameters
        ----------
        data_file : string
            The path to the sequences.
        vocab : string 

        Returns
        ----------
        X : list
            list of sequences where characters have been mapped to numbers.
        Y : list
            list of numerical labels (not one-hot)
        """
        assert vocab.lower() in ['dna', 'protein', 'arabic', 'inferred']
        X, Y = [], []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                seq, label = line.rstrip().split('\t')
                assert len(label) == 3
                if label in self._labels_to_use:
                    if len(seq) < 10:
                        continue
                    seq = list(seq)
                    seq = [self._vocab.add(token) for token in seq]
                    X.append(seq)
                    Y.append(self._classes.add(label))
            assert len(X) == len(Y)

        return X, Y

class DslUtility():
    def __init__(self, vocab=None):
        self._vocab = Vocabulary() if vocab is None else vocab
        self._classes = Vocabulary()

    def read_data(self, data_file, vocab='inferred'):
        assert vocab.lower() in ['dna', 'protein', 'arabic', 'inferred']
        X, Y = [], []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                seq, label = line.rstrip().split('\t')
                if len(seq) < 10:
                    continue
                seq = list(seq)
                seq = [self._vocab.add(token) for token in seq]
                X.append(seq)
                Y.append(self._classes.add(label))
            assert len(X) == len(Y)

        return X, Y

class FastskRunner():
    def __init__(self, prefix, data_location='/localtmp/dcb7xz/FastSK/data'):
        self.prefix = prefix
        self.train_file = osp.join(data_location, prefix + '.train.fasta')
        self.test_file = osp.join(data_location, prefix + '.test.fasta')
        
        reader = FastaUtility()
        self.Xtrain, self.Ytrain = reader.read_data(self.train_file)
        Xtest, Ytest = reader.read_data(self.test_file)
        Ytest = np.array(Ytest).reshape(-1, 1)
        self.Xtest, self.Ytest = Xtest, Ytest

    def compute_train_kernel(self, g, m, t=20, approx=True, I=100, delta=0.025, skip_variance=False):
        kernel = Kernel(g=g, m=m, t=t, approx=approx, max_iters=I, delta=delta, skip_variance=skip_variance)
        kernel.compute_train(self.Xtrain)

    def train_and_test(self, g, m, t, approx, I, delta=0.025, skip_variance=False, C=1):
        kernel = Kernel(g=g, m=m, t=t, approx=approx, max_iters=I, delta=delta, skip_variance=skip_variance)
        kernel.compute(self.Xtrain, self.Xtest)
        self.Xtrain = kernel.train_kernel()
        self.Xtest = kernel.test_kernel()
        svm = LinearSVC(C=C, class_weight='balanced')
        self.clf = CalibratedClassifierCV(svm, cv=5).fit(self.Xtrain, self.Ytrain)
        acc, auc = self.evaluate_clf()
        return acc, auc

    def evaluate_clf(self):
        acc = self.clf.score(self.Xtest, self.Ytest)
        probs = self.clf.predict_proba(self.Xtest)[:,1]
        auc = metrics.roc_auc_score(self.Ytest, probs)
        return acc, auc

class GkmRunner():
    def __init__(self, exec_location, data_locaton, dataset, outdir="./temp"):
        r"""Initialize a GkmRunner object for running gkm-SVM2.9
        
        Parameters
        ----------
        exec_location : string
            folder containing gkm-SVM2.0 executables
        data_location : string
            folder containing data files in the gkm-SVM2.0 format
        dataset : string
            name of the dataset to use
        outdir : string 
            name of directory to save files created by gkm-SVM2.0
        """
        self.exec_location = exec_location
        self.dir = data_locaton
        self.dataset = dataset
        self.outdir = outdir
        self.g, self.k, self.approx = 0, 0, False

        ## Data files
        self.train_pos_file = osp.join(self.dir, self.dataset + '.train.pos.fasta')
        self.train_neg_file = osp.join(self.dir, self.dataset + '.train.neg.fasta')
        self.test_pos_file = osp.join(self.dir, self.dataset + '.test.pos.fasta')
        self.test_neg_file = osp.join(self.dir, self.dataset + '.test.neg.fasta')
        self.train_test_pos_file = osp.join(self.outdir, self.dataset + '.train_test.pos.fasta')
        self.train_test_neg_file = osp.join(self.outdir, self.dataset + '.train_test.neg.fasta')
        
        ## Temp files that gkm creates
        if not osp.exists(self.outdir):
            os.makedirs(self.outdir)
        self.train_kernel_file = osp.join(self.outdir, self.dataset + '_train_kernel.out')
        self.test_kernel_file = osp.join(self.outdir, self.dataset + '_train_kernel.out')
        self.train_test_kernel_file = osp.join(self.outdir, self.dataset + '_train_test_kernel.out')
        self.svm_file_prefix = osp.join(self.outdir, "svmtrain")
        self.svmalpha = self.svm_file_prefix + '_svalpha.out'
        self.svseq = self.svm_file_prefix + '_svseq.fa'
        self.pos_pred_file = osp.join(self.outdir, self.dataset + '.preds.pos.out')
        self.neg_pred_file = osp.join(self.outdir, self.dataset + '.preds.neg.out')

        ## Vals for train-test kernel
        self.num_pos_train, self.num_neg_train, self.num_pos_test, self.num_neg_test = [0] * 4   
        self.kernel_tri = []

    def compute_kernel(self, g, m, t, approx=False, alphabet=None, mode='train'):
        r"""Compute the training kernel using gkm-SVM2.0. The kernel function is given by:
        .. math::
            K_{gkm}(x,y) = \sum_{d=0}^{g}N_d(x,y)h_d
        
        where :math:`N_d(x,y)` denotes the d-mismatch neighborhood of $x$, $y$. I.e., the number of pairs of $g$-mers
        with a Hamming distance of up to $d$ shared in common between samples $x$ and $y$.

        Parameters
        ----------
        g : int
            word length
        m : int
            number of gaps/mismatch positions. Note k = g - m
        t : int
            number of threads to use
        approx : boolean
            whether to perform the truncated summation of gkm-SVM2.0 algorithm.

            In particular, this will set the '-d' flag to 3 (the default)
        dataset : string
            name of the dataset to use
        outdir : string 
            name of directory to save files created by gkm-SVM2.0

        Returns
        ----------
        X : list
            list of sequences where characters have been mapped to numbers.
        Y : list
            list of labels
        """
        assert mode in ['train', 'test', 'train_test']
        k = g - m
        self.g, self.k, self.approx = g, k, approx

        ### compute kernel ###
        execute = osp.join(self.exec_location, 'gkmsvm_kernel')
        command = [execute,
            '-a', str(2),
            '-l', str(g),
            '-k', str(k),
            '-T', str(t),
            '-R']
        if (not approx and m > 0):
            command += ['-d', str(m)]
        else:
            command += ['-d', str(3)]
        if alphabet is not None:
            command += ['-A', alphabet]
        if mode == 'train':
            command += [self.train_pos_file, self.train_neg_file, self.train_kernel_file]
        elif mode == 'test':
            command += [self.test_pos_file, self.test_neg_file, self.test_kernel_file]
        else:
            command += [self.train_test_pos_file, self.train_test_neg_file, self.train_test_kernel_file]
        print(' '.join(command))
        output = subprocess.check_output(command)

    def train_and_test(self, g, m, t, approx=False, alphabet=None):
        self.combine_train_and_test()
        self.compute_kernel(g, m, t, approx=approx, alphabet=alphabet, mode='train_test')
        self.parse_train_test_kernel()

    def parse_train_test_kernel(self):
        Xtrain, Ytrain, Xtest, Ytest = [], [], [], []
        self.kernel_tri
        num_train = self.num_pos_train + self.num_neg_train
        with open(self.train_test_kernel_file, 'r') as f:
            for line in f:
                vals = line.strip().split('\t')
                self.kernel_tri.append(vals)

        '''kernel file:
            - train_pos
            - test_pos
            - train_neg
            - test_neg
        '''

        # pos train
        start = 0
        for i in range(self.num_pos_train):
            x = []
            for j in range(num_train):
                x.append(self.tri_access(i, j))
                start += 1
            assert len(x) == num_train
            Xtrain.append(x)
            Ytrain.append(1)

        # test pos
        for i in range(start, start + self.num_pos_test):
            x = []
            for j in range(num_train):
                x.append(self.tri_access(i , j))
                start += 1
            assert len(x) == num_train
            Xtest.append(x)
            Ytest.append(1)

        # train neg
        for i in range(start, start + self.num_neg_train):
            x = []
            for j in range(num_train):
                x.append(self.tri_access(i, j))
                start += 1
            assert len(x) == num_train
            Xtrain.append(x)
            Ytrain.append(-1)

        # test neg
        for i in range(start, start + self.num_neg_test):
            x = []
            for j in range(num_train):
                x.append(self.tri_access(i, j))
                start += 1
            assert len(x) == num_train
            Xtest.append(x)
            Ytest.append(-1)

        print(len(tri[0]))

    def tri_access(self, i, j):
        if j > i:
            i, j = j, i
        return self.kernel_tri[i][j]

    def combine_train_and_test(self):
        pos_sequences, neg_sequences = [], []
        pos_labels, neg_labels = [], []
        num_pos_train, num_neg_train = 0, 0
        num_pos_test, num_neg_test = 0, 0

        with open(self.train_pos_file, 'r') as f:
            label_line = True
            for line in f:
                line = line.rstrip()
                if label_line:
                    pos_labels.append(line)
                    label_line = False
                else:
                    pos_sequences.append(line)
                    self.num_pos_train += 1
                    label_line = True

        with open(self.test_pos_file, 'r') as f:
            label_line = True
            for line in f:
                line = line.rstrip()
                if label_line:
                    pos_labels.append(line)
                    label_line = False
                else:
                    pos_sequences.append(line)
                    self.num_pos_test += 1
                    label_line = True

        with open(self.train_neg_file, 'r') as f:
            label_line = True
            for line in f:
                line = line.rstrip()
                if label_line:
                    neg_labels.append(line)
                    label_line = False
                else:
                    neg_sequences.append(line)
                    self.num_neg_train += 1
                    label_line = True

        with open(self.test_neg_file, 'r') as f:
            label_line = True
            for line in f:
                line = line.rstrip()
                if label_line:
                    neg_labels.append(line)
                    label_line = False
                else:
                    neg_sequences.append(line)
                    self.num_neg_test += 1
                    label_line = True

        with open(self.train_test_pos_file, 'w+') as f:
            for seq, label in zip(pos_sequences, pos_labels):
                f.write('{}\n{}\n'.format(label, seq))

        with open(self.train_test_neg_file, 'w+') as f:
            for seq, label in zip(neg_sequences, neg_labels):
                f.write('{}\n{}\n'.format(label, seq))


class GaKCoRunner():
    def __init__(self, exec_location, data_locaton, type_, prefix, outdir='./temp'):
        self.exec_location = exec_location
        self.data_locaton = data_locaton
        self.train_file = osp.join('/localtmp/dcb7xz/FastSK/data', prefix + '.train.fasta')
        self.test_file = osp.join('/localtmp/dcb7xz/FastSK/data', prefix + '.test.fasta')
        self.train_test_file = osp.join(outdir, prefix + '_train_test.fasta')
        assert type_ in ['dna', 'protein']
        if type_ == 'protein':
            self.dict_file = osp.join(data_locaton, 'full_prot.dict.txt')
        else:
            self.dict_file = osp.join(data_locaton, 'dna.dictionary.txt')
        self.labels_file = osp.join(outdir, 'labels.txt')
        self.kernel_file = osp.join(outdir, 'kernel.txt')
        self.num_train, self.num_test = 0, 0

    def compute_kernel(self, g, m, mode='train', t=1):
        self.g = g
        self.m = m
        self.k = g - m

        assert mode in ['train', 'test', 'train_test']
        if mode == 'train':
            data_file = self.train_file
        elif mode == 'test':
            data_file = self.test_file
        else:
            data_file = self.train_test_file

        command = [self.exec_location,
            '-g', str(self.g),
            '-k', str(self.k),
            data_file,
            self.dict_file,
            self.labels_file,
            self.kernel_file]
            
        output = subprocess.check_output(command)

    def train_and_test(self, g, m, C=1):
        self.combine_train_and_test()
        self.compute_kernel(g, m, mode='train_test')

        self.Xtrain, self.Xtest = self.read_kernel()
        self.Ytrain, self.Ytest = self.read_labels()
        
        svm = LinearSVC(C=C)
        self.clf = CalibratedClassifierCV(svm, cv=5).fit(self.Xtrain, self.Ytrain)
        acc, auc = self.evaluate_clf()
        return acc, auc

    def evaluate_clf(self):
        acc = self.clf.score(self.Xtest, self.Ytest)
        probs = self.clf.predict_proba(self.Xtest)[:,1]
        auc = metrics.roc_auc_score(self.Ytest, probs)
        return acc, auc

    def combine_train_and_test(self):
        lines = []
        with open(self.train_file, 'r') as f:
            for line in f:
                if line[0] == '>':
                    self.num_train += 1
                lines.append(line)
        with open(self.test_file, 'r') as f:
            for line in f:
                if line[0] == '>':
                    self.num_test += 1
                lines.append(line)
        with open(self.train_test_file, 'w+') as f:
            f.writelines(lines)

    def read_labels(self):
        Ytrain, Ytest = [], []
        with open(self.train_file, 'r') as f:
            for line in f:
                if line[0] == '>':
                    Ytrain.append(line.rstrip().split('>')[1])
        with open(self.test_file, 'r') as f:
            for line in f:
                if line[0] == '>':
                    Ytest.append(line.rstrip().split('>')[1])

        return Ytrain, Ytest

    def read_kernel(self):
        Xtrain, Xtest = [], []
        with open(self.kernel_file, 'r') as f:
            count = 0
            for line in f:
                x = [float(item.split(':')[1]) for item in line.rstrip().split(' ')][:self.num_train]
                if (count < self.num_train):
                    Xtrain.append(x)
                else:
                    Xtest.append(x)
                count += 1

        return Xtrain, Xtest

    def get_labels(self):
        pass

class BlendedSpectrumRunner():
    def __init__(self, exec_dir, data_locaton, prefix, outdir="./temp"):
        self.exec_dir = exec_dir
        self.train_fasta = osp.join(data_locaton, prefix + '.train.fasta')
        self.test_fasta = osp.join(data_locaton, prefix + '.test.fasta')
        self.outdir = outdir
        if not osp.exists(self.outdir):
            os.makedirs(self.outdir)
        self.train_seq = osp.join(self.outdir, prefix + '_spectrum.train.txt')
        self.test_seq = osp.join(self.outdir, prefix + '_spectrum.test.txt')
        self.train_and_test_seq = osp.join(self.outdir, prefix + '_.train-tst.spectrum.txt')
        self.num_train, self.num_test = 0, 0
        self.write_seq(self.train_fasta, mode='train')

        self.kernel_file = osp.join(outdir, "kernel.txt")

    def combine_train_and_test(self):
        Xtrain, Xtest, self.Ytrain, self.Ytest = [], [], [], []
        with open(self.train_fasta, 'r') as f:
            label_line = True
            for line in f:
                line = line.rstrip()
                if label_line:
                    self.num_train += 1
                    label = line.split('>')[1]
                    self.Ytrain.append(label)
                    label_line = False
                else:
                    Xtrain.append(line.lower())
                    label_line = True
        with open(self.test_fasta, 'r') as f:
            label_line = True
            for line in f:
                line = line.rstrip()
                if label_line:
                    self.num_test += 1
                    label = line.split('>')[1]
                    self.Ytest.append(label)
                    label_line = False
                else:
                    Xtest.append(line.lower())
                    label_line = True
        X = Xtrain + Xtest
        with open(self.train_and_test, 'w+') as f:
            for x in X:
                f.write(x + '\n')

    
    def write_seq(self, datafile, mode='train'):        
        assert mode in ['train', 'test']
        if mode == 'train':
            outfile = self.train_seq
        else:
            outfile = self.test_seq
        X, Y = [], []
        with open(datafile, 'r') as f:
            label_line = True
            for line in f:
                line = line.rstrip()
                if label_line:
                    label = line.split('>')[1]
                    Y.append(label)
                    label_line = False
                else:
                    X.append(line.lower())
                    label_line = True
        
        with open(outfile, 'w+') as f:
            for x in X:
                f.write(x + '\n')

    def compute_kernel(self, k1=3, k2=5, mode='train_and_test'):
        self.k1, self.k2 = k1, k2

        datafile = self.train_seq
        assert mode in ['train', 'test', 'train_and_test']
        if mode == 'train':
            datafile = self.train_seq
        elif mode == 'test':
            datafile = self.test_seq
        else:
            datafile = self.train_and_test_seq

        command = ["java",
            '-cp', self.exec_dir,
            'ComputeStringKernel',
            "spectrum",
            str(self.k1),
            str(self.k2),
            datafile,
            self.kernel_file]
        output = subprocess.check_output(command)

    def read_kernel(self):
        Xtrain, Xtest = [], []
        with open(self.kernel_file, 'r') as f:
            count = 0
            for line in f:
                x = [float(item) for item in line.rstrip().split(' ')][:self.num_train]
                if (count < self.num_train):
                    Xtrain.append(x)
                else:
                    Xtest.append(x)
                count += 1

        return Xtrain, Xtest

    def train_and_test(self, k1=3, k2=5, C=1):
        self.combine_train_and_test()
        self.compute_kernel(k1, k2, mode='train_and_test')

        self.Xtrain, self.Xtest = self.read_kernel()
        
        svm = LinearSVC(C=C, class_weight='balanced', max_iter=3000)
        self.clf = CalibratedClassifierCV(svm, cv=5).fit(self.Xtrain, self.Ytrain)
        acc, auc = self.evaluate_clf()
        return acc, auc

    def evaluate_clf(self):
        acc = self.clf.score(self.Xtest, self.Ytest)
        probs = self.clf.predict_proba(self.Xtest)[:,1]
        auc = metrics.roc_auc_score(self.Ytest, probs)
        return acc, auc

