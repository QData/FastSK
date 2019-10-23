import os
import os.path as osp
import subprocess

'''Utilities for demoing iGakco-SVM
'''

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

class GkmRunner():
    def __init__(self, exec_location, data_locaton, prefix, outdir="./temp"):
        self.exec_location = exec_location
        self.dir = data_locaton
        self.prefix = prefix
        self.outdir = outdir

        ## Data files
        self.train_pos_file = osp.join(self.dir, self.prefix + '.train.pos.fasta')
        self.train_neg_file = osp.join(self.dir, self.prefix + '.train.neg.fasta')
        self.test_pos_file = osp.join(self.dir, self.prefix + '.test.pos.fasta')
        self.test_neg_file = osp.join(self.dir, self.prefix + '.test.neg.fasta')
        
        ## Temp files that gkm creates
        if not osp.exists(self.outdir):
            os.makedirs(self.outdir)
        self.kernel_file = osp.join(self.outdir, self.prefix + '_kernel.out')
        self.svm_file_prefix = osp.join(self.outdir, "svmtrain")
        self.svmalpha = self.svm_file_prefix + '_svalpha.out'
        self.svseq = self.svm_file_prefix + '_svseq.fa'
        self.pos_pred_file = osp.join(self.outdir, self.prefix + '.preds.pos.out')
        self.neg_pred_file = osp.join(self.outdir, self.prefix + '.preds.neg.out')    

    def compute_kernel(self, g, m, t):
        k = g - m
        ### compute kernel ###
        execute = osp.join(self.exec_location, 'gkmsvm_kernel')
        command = [execute,
            '-a', str(2),
            '-l', str(g), 
            '-k', str(k), 
            '-d', str(m),
            '-T', str(t),
            '-R']
        command += [self.train_pos_file, self.train_neg_file, self.kernel_file]
        print(' '.join(command))
        output = subprocess.check_output(command)

class BlendedSpectrumRunner():
    def __init__(self, exec_location, data_locaton, prefix, outdir="./temp"):
        self.exec_location = exec_location
        self.train_data = "file.txt"
        self.kernel_file = osp.join(outdir, "kernel.txt")

    def compute_kernel(self, k1, k2):
        self.k1, self.k2 = k1, k2

        command = ["java", 
            self.exec_location, 
            "spectrum",
            str(self.k1),
            str(self.k2),
            self.train_data,
            self.kernel_file]
        output = subprocess.check_output(command)
