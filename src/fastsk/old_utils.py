import os
import os.path as osp
import subprocess
import numpy as np
from fastsk import FastSK
from sklearn.svm import LinearSVC
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn import metrics
import time
import multiprocessing
import subprocess


def time_fastsk(
    g,
    m,
    t,
    data_location,
    prefix,
    approx=False,
    max_iters=None,
    timeout=None,
    skip_variance=False,
):
    """Run FastSK kernel computation. If a timeout is provided,
    it'll run as a subprocess, which will be killed when the timeout is
    reached.
    """
    fastsk = FastskRunner(prefix, data_location)

    start = time.time()
    if timeout:
        if max_iters:
            args = {
                "t": t,
                "approx": approx,
                "skip_variance": skip_variance,
                "I": max_iters,
            }
        else:
            args = {"t": t, "approx": approx, "skip_variance": skip_variance}
        p = multiprocessing.Process(
            target=fastsk.compute_train_kernel,
            name="TimeFastSK",
            args=(g, m),
            kwargs=args,
        )
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            p.join()
    else:
        if max_iters:
            fastsk.compute_train_kernel(
                g, m, t=t, approx=approx, I=max_iters, skip_variance=skip_variance
            )
        else:
            fastsk.compute_train_kernel(
                g, m, t=t, approx=approx, skip_variance=skip_variance
            )

    end = time.time()

    return end - start


def fastsk_wrap(dataset, g, m, t, approx, I, delta, skip_variance, C, return_dict):
    fastsk = FastskRunner(dataset)
    acc, auc = fastsk.train_and_test(g, m, t, I, approx, skip_variance, C)
    return_dict["acc"] = acc
    return_dict["auc"] = auc


def train_and_test_fastsk(
    dataset, g, m, t, approx, I=50, delta=0.025, skip_variance=False, C=1, timeout=None
):
    start = time.time()
    if timeout:
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        p = multiprocessing.Process(
            target=fastsk_wrap,
            name="Train and test FastSK",
            args=(dataset, g, m, t, approx, I, delta, skip_variance, C, return_dict),
        )
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            p.join()

        if return_dict.values() == []:
            acc, auc = 0, 0
        else:
            acc, auc = return_dict["acc"], return_dict["auc"]
    else:
        acc, auc = fastsk.train_and_test(
            g, m, t=t, I=I, approx=approx, skip_variance=skip_variance, C=C
        )
    end = time.time()

    return acc, auc, end - start


def gkm_wrap(
    g, m, t, prefix, gkm_data, gkm_exec, approx, timeout, alphabet, return_dict
):
    k = g - m
    gkm = GkmRunner(gkm_exec, gkm_data, prefix, g, k, approx, alphabet, "./temp")
    acc, auc = gkm.train_and_test(t)
    return_dict["acc"] = acc
    return_dict["auc"] = auc


def train_and_test_gkm(
    g, m, t, prefix, gkm_data, gkm_exec, approx=False, timeout=None, alphabet=None
):
    start = time.time()
    if timeout:
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        p = multiprocessing.Process(
            target=gkm_wrap,
            name="Train and test Gkm",
            args=(
                g,
                m,
                t,
                prefix,
                gkm_data,
                gkm_exec,
                approx,
                timeout,
                alphabet,
                return_dict,
            ),
        )
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            p.join()
        if return_dict.values() == []:
            acc, auc = 0, 0
        else:
            acc, auc = return_dict["acc"], return_dict["auc"]
    else:
        k = g - m
        gkm = GkmRunner(gkm_exec, gkm_data, prefix, g, k, approx, alphabet, "./temp")
        acc, auc = gkm.train_and_test(t)
    end = time.time()
    return acc, auc, end - start


def time_gkm(
    g, m, t, prefix, gkm_data, gkm_exec, approx=False, timeout=None, alphabet=None
):
    """Run gkm-SVM2.0 kernel computation. If a timeout is provided,
    it'll be run as a subprocess, which will be killed when the timeout is
    reached.
    """
    k = g - m
    gkm = GkmRunner(gkm_exec, gkm_data, prefix, g, k, approx, alphabet, "./temp")

    start = time.time()
    if timeout:
        p = multiprocessing.Process(
            target=gkm.compute_train_kernel, name="TimeGkm", args=(t,)
        )
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            p.join()
    else:
        gkm.compute_train_kernel(t)

    end = time.time()

    return end - start


def time_gakco(g, m, type_, prefix, timeout=None):
    gakco_exec = "./baselines/GaKCo-SVM/bin/GaKCo"
    data = "./data/"
    gakco = GaKCoRunner(gakco_exec, data, type_, prefix)

    start = time.time()
    gakco.compute_kernel(g, m, mode="train")
    end = time.time()

    return end - start


def time_blended(k1, k2, prefix, timeout=None):
    blended_exec = "./baselines/String_Kernels_Package/code/"
    data = "../data/"
    blended = BlendedSpectrumRunner(blended_exec, data, prefix)
    start = time.time()
    blended.compute_kernel(k1, k2, mode="train")
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
        """Return the number tokens in the vocabulary."""
        return self._size

    def __str__(self):
        return str(self._token2idx)


class FastaUtility:
    def __init__(self, vocab=None):
        r"""
        Initialize a helper object for parsing datasets in FASTA-like format.

        Parameters
        ----------
        vocab :
        """
        self._vocab = Vocabulary() if vocab is None else vocab

    def read_data(self, data_file, vocab="inferred", regression=False):
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
        assert vocab.lower() in ["dna", "protein", "inferred"]
        X, Y = [], []
        with open(data_file, "r") as f:
            label_line = True
            for line in f:
                line = line.strip().lower()
                if label_line:
                    split = line.split(">")
                    assert len(split) == 2
                    if regression:
                        label = split[1]
                    else:
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


class ArabicUtility:
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
        self._labels_to_use = ["RAB", "BEI", "DOH", "CAI", "TUN", "MSA"]

    def read_data(self, data_file, vocab="inferred"):
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
        assert vocab.lower() in ["dna", "protein", "arabic", "inferred"]
        X, Y = [], []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                seq, label = line.rstrip().split("\t")
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


class DslUtility:
    def __init__(self, vocab=None):
        self._vocab = Vocabulary() if vocab is None else vocab
        self._classes = Vocabulary()

    def read_data(self, data_file, vocab="inferred"):
        assert vocab.lower() in ["dna", "protein", "arabic", "inferred"]
        X, Y = [], []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                seq, label = line.rstrip().split("\t")
                if len(seq) < 10:
                    continue
                seq = list(seq)
                seq = [self._vocab.add(token) for token in seq]
                X.append(seq)
                Y.append(self._classes.add(label))
            assert len(X) == len(Y)

        return X, Y


class FastskRunner:
    def __init__(self, prefix, data_location="../data"):
        self.prefix = prefix
        self.train_file = osp.join(data_location, prefix + ".train.fasta")
        self.test_file = osp.join(data_location, prefix + ".test.fasta")

        reader = FastaUtility()
        self.train_seq, self.Ytrain = reader.read_data(self.train_file)
        self.test_seq, Ytest = reader.read_data(self.test_file)
        Ytest = np.array(Ytest).reshape(-1, 1)
        self.Ytest = Ytest

    def compute_train_kernel(
        self, g, m, t=20, approx=True, I=100, delta=0.025, skip_variance=False
    ):
        kernel = FastSK(
            g=g,
            m=m,
            t=t,
            approx=approx,
            max_iters=I,
            delta=delta,
            skip_variance=skip_variance,
        )
        kernel.compute_train(self.train_seq)

    def train_and_test(
        self, g, m, t, approx, I=100, delta=0.025, skip_variance=False, C=1
    ):
        kernel = FastSK(
            g=g,
            m=m,
            t=t,
            approx=approx,
            max_iters=I,
            delta=delta,
            skip_variance=skip_variance,
        )

        kernel.compute_kernel(self.train_seq, self.test_seq)
        self.Xtrain = kernel.get_train_kernel()
        self.Xtest = kernel.get_test_kernel()
        self.stdevs = kernel.get_stdevs()
        svm = LinearSVC(C=C, class_weight="balanced")
        self.clf = CalibratedClassifierCV(svm, cv=5).fit(self.Xtrain, self.Ytrain)
        acc, auc = self.evaluate_clf()
        return acc, auc

    def evaluate_clf(self):
        acc = self.clf.score(self.Xtest, self.Ytest)
        probs = self.clf.predict_proba(self.Xtest)[:, 1]
        auc = metrics.roc_auc_score(self.Ytest, probs)
        return acc, auc


class FastskRegressor:
    def __init__(self, dataset, data_location="../data"):
        self.dataset = dataset
        self.train_file = osp.join(data_location, prefix + ".train.fasta")
        self.test_file = osp.join(data_location, prefix + ".test.fasta")

        reader = FastaUtility()
        self.train_seq, self.Ytrain = reader.read_data(train_file, regression=True)
        self.test_seq, self.Ytest = reader.read_data(test_file, regression=True)
        self.Ytrain = np.array(self.Ytrain).astype(np.float)
        self.Ytest = np.array(self.Ytest).astype(np.float)

    def compute_train_kernel(
        self, g, m, t=20, approx=True, I=100, delta=0.025, skip_variance=False
    ):
        kernel = FastSK(
            g=g,
            m=m,
            t=t,
            approx=approx,
            max_iters=I,
            delta=delta,
            skip_variance=skip_variance,
        )
        kernel.compute_train(self.train_seq)

    def train_and_test(self, g, m, t, approx, I=100, delta=0.025, skip_variance=False):
        kernel = FastSK(
            g=g,
            m=m,
            t=t,
            approx=approx,
            max_iters=I,
            delta=delta,
            skip_variance=skip_variance,
        )

        kernel.compute_kernel(self.train_seq, self.test_seq)
        self.Xtest = kernel.get_test_kernel()
        self.Xtest = np.array(self.Xtest).reshape(len(self.Xtest), -1)
        self.Xtrain = kernel.get_train_kernel()
        self.Xtrain = np.array(self.Xtrain).reshape(len(self.Xtrain), -1)

        # Can replace Lasso with alternative regression approaches such as SVR
        model = LassoCV(cv=5, n_jobs=t, random_state=293).fit(self.Xtrain, self.Ytrain)
        r2 = model.score(self.Xtest, self.Ytest)
        return r2


class GkmRunner:
    def __init__(
        self,
        exec_location,
        data_locaton,
        dataset,
        g,
        k,
        approx=False,
        alphabet=None,
        outdir="./temp",
    ):
        self.exec_location = exec_location
        self.dir = data_locaton
        self.dataset = dataset
        self.outdir = outdir
        self.g, self.k, self.alphabet = g, k, alphabet

        """Important note:
        gkmSVM's -d parameter (max_m) is *not* the same as our
        m = g - k parameter. It's actually the upper bound of the
        summation shown in equation 3 in the
        2014 gkmSVM paper (ghandi2014enhanced)."""
        if approx:
            """By default, their approximation algorithm truncates the
            summation from eq. 3 to a value of 3 mismatches.
            """
            self.max_m = 3
        else:
            """If using the exact algo, the summation runs from
            0 to l (their l is our g)
            """
            self.max_m = self.g

        ## Data files
        self.train_pos_file = osp.join(self.dir, self.dataset + ".train.pos.fasta")
        self.train_neg_file = osp.join(self.dir, self.dataset + ".train.neg.fasta")
        self.test_pos_file = osp.join(self.dir, self.dataset + ".test.pos.fasta")
        self.test_neg_file = osp.join(self.dir, self.dataset + ".test.neg.fasta")
        self.train_test_pos_file = osp.join(
            self.outdir, self.dataset + ".train_test.pos.fasta"
        )
        self.train_test_neg_file = osp.join(
            self.outdir, self.dataset + ".train_test.neg.fasta"
        )

        ## Temp files that gkm creates
        if not osp.exists(self.outdir):
            os.makedirs(self.outdir)
        self.kernel_file = osp.join(self.outdir, self.dataset + "_kernel.out")
        self.svm_file_prefix = osp.join(self.outdir, "svmtrain")
        self.svmalpha = self.svm_file_prefix + "_svalpha.out"
        self.svseq = self.svm_file_prefix + "_svseq.fa"
        self.pos_pred_file = osp.join(self.outdir, self.dataset + ".preds.pos.out")
        self.neg_pred_file = osp.join(self.outdir, self.dataset + ".preds.neg.out")

    def compute_train_kernel(self, t):
        execute = osp.join(self.exec_location, "gkmsvm_kernel")
        command = [
            execute,
            "-a",
            str(2),
            "-l",
            str(self.g),
            "-k",
            str(self.k),
            "-d",
            str(self.max_m),
            "-T",
            str(t),
            "-R",
        ]
        if self.alphabet is not None:
            command += ["-A", self.alphabet]
        command += [self.train_pos_file, self.train_neg_file, self.kernel_file]
        print(" ".join(command))
        output = subprocess.check_output(command)

    def train_svm(self):
        execute = osp.join(self.exec_location, "gkmsvm_train")
        command = [
            execute,
            self.kernel_file,
            self.train_pos_file,
            self.train_neg_file,
            self.svm_file_prefix,
        ]
        print(" ".join(command))
        output = subprocess.check_output(command)

    def classify(self):
        ## pos predictions
        execute = osp.join(self.exec_location, "gkmsvm_classify")
        command = [
            execute,
            "-l",
            str(self.g),
            "-k",
            str(self.k),
            "-d",
            str(self.max_m),
            "-R",
        ]
        if self.alphabet is not None:
            command += ["-A", self.alphabet]
        command += [self.test_pos_file, self.svseq, self.svmalpha, self.pos_pred_file]
        print(" ".join(command))
        subprocess.check_output(command)

        # get neg preds
        command = [
            execute,
            "-l",
            str(self.g),
            "-k",
            str(self.k),
            "-d",
            str(self.max_m),
            "-R",
        ]
        if self.alphabet is not None:
            command += ["-A", self.alphabet]

        command += [self.test_neg_file, self.svseq, self.svmalpha, self.neg_pred_file]
        print(" ".join(command))
        subprocess.check_output(command)

    def evaluate(self):
        pos_preds = self.read_preds(self.pos_pred_file)
        neg_preds = self.read_preds(self.neg_pred_file)

        print("Computing accuracy...")
        acc = self.get_accuracy(pos_preds, neg_preds)
        print("Computing AUC...")
        auc = self.get_auc(pos_preds, neg_preds)
        print("Accuracy = {}, AUC = {}".format(acc, auc))
        return acc, auc

    def train_and_test(self, t=20):
        self.compute_train_kernel(t)
        self.train_svm()
        self.classify()
        acc, auc = self.evaluate()
        return acc, auc

    def read_preds(self, file):
        preds = []
        with open(file, "r") as f:
            for line in f:
                line = line.split()
                assert len(line) == 2
                preds.append(float(line[1]))
        return preds

    def get_accuracy(self, pos_preds, neg_preds):
        accuracy = 0
        num_correct = 0
        num_pred = len(pos_preds) + len(neg_preds)
        for pred in pos_preds:
            if pred > 0:
                num_correct += 1
        for pred in neg_preds:
            if pred <= 0:
                num_correct += 1
        return num_correct / num_pred

    def get_auc(self, pos_preds, neg_preds):
        ytrue = [1 for _ in pos_preds] + [-1 for _ in neg_preds]
        yscore = [score for score in pos_preds] + [score for score in neg_preds]
        auc = metrics.roc_auc_score(ytrue, yscore)
        return auc


class GaKCoRunner:
    def __init__(self, exec_location, data_locaton, type_, prefix, outdir="./temp"):
        self.exec_location = exec_location
        self.data_locaton = data_locaton
        self.train_file = osp.join("../data", prefix + ".train.fasta")
        self.test_file = osp.join("../data", prefix + ".test.fasta")
        self.train_test_file = osp.join(outdir, prefix + "_train_test.fasta")
        assert type_ in ["dna", "protein"]
        if type_ == "protein":
            self.dict_file = osp.join(data_locaton, "full_prot.dict.txt")
        else:
            self.dict_file = osp.join(data_locaton, "dna.dictionary.txt")
        self.labels_file = osp.join(outdir, "labels.txt")
        self.kernel_file = osp.join(outdir, "kernel.txt")
        self.num_train, self.num_test = 0, 0

    def compute_kernel(self, g, m, mode="train", t=1):
        self.g = g
        self.m = m
        self.k = g - m

        assert mode in ["train", "test", "train_test"]
        if mode == "train":
            data_file = self.train_file
        elif mode == "test":
            data_file = self.test_file
        else:
            data_file = self.train_test_file

        command = [
            self.exec_location,
            "-g",
            str(self.g),
            "-k",
            str(self.k),
            data_file,
            self.dict_file,
            self.labels_file,
            self.kernel_file,
        ]

        output = subprocess.check_output(command)

    def train_and_test(self, g, m, C=1):
        self.combine_train_and_test()
        self.compute_kernel(g, m, mode="train_test")

        self.Xtrain, self.Xtest = self.read_kernel()
        self.Ytrain, self.Ytest = self.read_labels()

        svm = LinearSVC(C=C)
        self.clf = CalibratedClassifierCV(svm, cv=5).fit(self.Xtrain, self.Ytrain)
        acc, auc = self.evaluate_clf()
        return acc, auc

    def evaluate_clf(self):
        acc = self.clf.score(self.Xtest, self.Ytest)
        probs = self.clf.predict_proba(self.Xtest)[:, 1]
        auc = metrics.roc_auc_score(self.Ytest, probs)
        return acc, auc

    def combine_train_and_test(self):
        lines = []
        with open(self.train_file, "r") as f:
            for line in f:
                if line[0] == ">":
                    self.num_train += 1
                lines.append(line)
        with open(self.test_file, "r") as f:
            for line in f:
                if line[0] == ">":
                    self.num_test += 1
                lines.append(line)
        with open(self.train_test_file, "w+") as f:
            f.writelines(lines)

    def read_labels(self):
        Ytrain, Ytest = [], []
        with open(self.train_file, "r") as f:
            for line in f:
                if line[0] == ">":
                    Ytrain.append(line.rstrip().split(">")[1])
        with open(self.test_file, "r") as f:
            for line in f:
                if line[0] == ">":
                    Ytest.append(line.rstrip().split(">")[1])

        return Ytrain, Ytest

    def read_kernel(self):
        Xtrain, Xtest = [], []
        with open(self.kernel_file, "r") as f:
            count = 0
            for line in f:
                x = [float(item.split(":")[1]) for item in line.rstrip().split(" ")][
                    : self.num_train
                ]
                if count < self.num_train:
                    Xtrain.append(x)
                else:
                    Xtest.append(x)
                count += 1

        return Xtrain, Xtest

    def get_labels(self):
        pass


class BlendedSpectrumRunner:
    def __init__(self, exec_dir, data_locaton, prefix, outdir="./temp"):
        self.exec_dir = exec_dir
        self.train_fasta = osp.join(data_locaton, prefix + ".train.fasta")
        self.test_fasta = osp.join(data_locaton, prefix + ".test.fasta")
        self.outdir = outdir
        if not osp.exists(self.outdir):
            os.makedirs(self.outdir)
        self.train_seq = osp.join(self.outdir, prefix + "_spectrum.train.txt")
        self.test_seq = osp.join(self.outdir, prefix + "_spectrum.test.txt")
        self.train_and_test_seq = osp.join(
            self.outdir, prefix + "_.train-tst.spectrum.txt"
        )
        self.num_train, self.num_test = 0, 0
        self.write_seq(self.train_fasta, mode="train")

        self.kernel_file = osp.join(outdir, "kernel.txt")

    def combine_train_and_test(self):
        Xtrain, Xtest, self.Ytrain, self.Ytest = [], [], [], []
        with open(self.train_fasta, "r") as f:
            label_line = True
            for line in f:
                line = line.rstrip()
                if label_line:
                    self.num_train += 1
                    label = line.split(">")[1]
                    self.Ytrain.append(label)
                    label_line = False
                else:
                    Xtrain.append(line.lower())
                    label_line = True
        with open(self.test_fasta, "r") as f:
            label_line = True
            for line in f:
                line = line.rstrip()
                if label_line:
                    self.num_test += 1
                    label = line.split(">")[1]
                    self.Ytest.append(label)
                    label_line = False
                else:
                    Xtest.append(line.lower())
                    label_line = True
        X = Xtrain + Xtest
        with open(self.train_and_test, "w+") as f:
            for x in X:
                f.write(x + "\n")

    def write_seq(self, datafile, mode="train"):
        assert mode in ["train", "test"]
        if mode == "train":
            outfile = self.train_seq
        else:
            outfile = self.test_seq
        X, Y = [], []
        with open(datafile, "r") as f:
            label_line = True
            for line in f:
                line = line.rstrip()
                if label_line:
                    label = line.split(">")[1]
                    Y.append(label)
                    label_line = False
                else:
                    X.append(line.lower())
                    label_line = True

        with open(outfile, "w+") as f:
            for x in X:
                f.write(x + "\n")

    def compute_kernel(self, k1=3, k2=5, mode="train_and_test"):
        self.k1, self.k2 = k1, k2

        datafile = self.train_seq
        assert mode in ["train", "test", "train_and_test"]
        if mode == "train":
            datafile = self.train_seq
        elif mode == "test":
            datafile = self.test_seq
        else:
            datafile = self.train_and_test_seq

        command = [
            "java",
            "-cp",
            self.exec_dir,
            "ComputeStringKernel",
            "spectrum",
            str(self.k1),
            str(self.k2),
            datafile,
            self.kernel_file,
        ]
        output = subprocess.check_output(command)

    def read_kernel(self):
        Xtrain, Xtest = [], []
        with open(self.kernel_file, "r") as f:
            count = 0
            for line in f:
                x = [float(item) for item in line.rstrip().split(" ")][: self.num_train]
                if count < self.num_train:
                    Xtrain.append(x)
                else:
                    Xtest.append(x)
                count += 1

        return Xtrain, Xtest

    def train_and_test(self, k1=3, k2=5, C=1):
        self.combine_train_and_test()
        self.compute_kernel(k1, k2, mode="train_and_test")

        self.Xtrain, self.Xtest = self.read_kernel()

        svm = LinearSVC(C=C, class_weight="balanced", max_iter=3000)
        self.clf = CalibratedClassifierCV(svm, cv=5).fit(self.Xtrain, self.Ytrain)
        acc, auc = self.evaluate_clf()
        return acc, auc

    def evaluate_clf(self):
        acc = self.clf.score(self.Xtest, self.Ytest)
        probs = self.clf.predict_proba(self.Xtest)[:, 1]
        auc = metrics.roc_auc_score(self.Ytest, probs)
        return acc, auc
