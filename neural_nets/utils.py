import random
from sklearn import metrics
import numpy as np

import torch
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

PAD_IDX = 0

def get_evaluation(y_true, y_prob):
    y_pred = np.argmax(y_prob, -1)
    pos_scores = y_prob[:,1].tolist()
    
    '''Exploding gradient can cause logits to
    contain NaN values, which will make roc_auc_score
    crash. Exploding gradient is avoided if:
        * learning rate < 1
        * we use gradient clipping (see torch.nn.utils.clip_grad_norm)
    '''
    try:
        auc = metrics.roc_auc_score(y_true, pos_scores)
    except ValueError as e:
        print("y_prob = ", y_prob)
        print("y_true = ", y_true)
        auc = 0
        exit()
    evaluation = {
        'accuracy': metrics.accuracy_score(y_true, y_pred),
        'auc': auc
    }
    return evaluation


def collate(batch):
    batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    sequences = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    lengths = [seq.shape[0] for seq in sequences]
    x = pad_sequence(sequences, padding_value=0, batch_first=False)
    y = torch.LongTensor(labels)
    return [x, y, lengths]

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Vocabulary(object):
    """A class for storing the vocabulary of a 
    sequence dataset. Maps words or characters to indexes in the
    vocabulary.
    """
    def __init__(self):
        self._token2idx = {}
        self._idx2token = {}
        self._token2idx[0] = 0
        self._idx2token[0] = 0
        self._size = 1

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
            self._token2idx[self._size] = token
            self._size += 1
        return self._token2idx.get(token)

    def size(self):
        """Return the number tokens in the vocabulary.
        """
        return self._size

    def __len__(self):
        return self.size()

class Fold(data.Dataset):
    """Dataset wrapper for a cross validation fold. Used to
    make cross validation sampling with DataLoader easy.
    Args:
        
    """
    def __init__(self, sequences, labels, transform=None):
        self.sequences = sequences
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        sequence = sequence if self.transform is None else self.transform(sequence)
        label = self.labels[idx]
        return sequence, label


class FastaDataset(data.Dataset):
    """Dataset class for creating FASTA-formatted
    sequence datasets.
    Args:
        file_path (string): path to the fasta file
        vocab (Vocabulary): a predefined vocabulary to use. Recommended if
            the dataset represents a test set that should have the exact
            same vocabulary as the training set.
        one_hot (bool): whether or not use one-hot encoding for the characters
            in the sequences. Sequences will have dimension (Sigma x length)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
    """

    def __init__(self, file_path, vocab=None, one_hot=False, transform=None):
        self.file_path = file_path
        self.transform = transform
        self._vocab = Vocabulary() if vocab is None else vocab
        self.one_hot = one_hot
        self.sequences = []
        self.padded_sequences = []
        self.labels = []
        self.max_length = 0
        self._read_data()
        self._folds = []
        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        #sequence = self.padded_sequences[idx]
        sequence = sequence if self.transform is None else self.transform(sequence)
        label = self.labels[idx]
        return sequence, label
    
    def get_vocab(self):
        return self._vocab

    def split(self, k=5):
        """Shuffle dataset and split into k folds.
        Args:
            k (int): number of 
        """

        # unison shuffle sequences and labels
        shuffle = list(zip(self.sequences, self.labels))
        random.shuffle(shuffle)
        self.sequences, self.labels = zip(*shuffle)

        # create folds
        total_size = len(self.sequences)
        fold_size = total_size // k
        for i in range(k):
            start = i * fold_size
            end = start + fold_size if i < k - 1 else total_size
            fold_sequences = self.sequences[start:end]
            fold_labels = self.labels[start:end]
            self._folds.append(list(zip(fold_sequences, fold_labels)))

    def _read_data(self):
        """Read a file in FASTA format. Specifically, resembles:
                >0
                ATCG
            where the first line is assumed to be a label line.
            Will read from self._file_path and store sequences in 
            self.sequences and labels in self.labels.
        """
        with open(self.file_path, 'r', encoding='utf-8') as f:
            label_line = True
            for line in f:
                line = line.strip().lower()
                if label_line:
                    split = line.split('>')
                    assert len(split) == 2
                    label = int(split[1])
                    assert label in [-1, 0, 1]
                    label = torch.tensor([label], dtype=torch.long)
                    self.labels.append(label)
                    label_line = False
                else:
                    seq = list(line)
                    seq = [self._vocab.add(token) for token in seq]
                    self.max_length = max(self.max_length, len(seq))
                    seq = torch.tensor(seq, dtype=torch.long)
                    if self.one_hot:
                        seq = F.one_hot(seq).T
                    self.sequences.append(seq)
                    label_line = True

        assert len(self.sequences) == len(self.labels)

    def get_fold(self, idx):
        """Return 2 dataloaders where the ith fold is a
        validation set.
        """
        num_folds = len(self._folds)
        if (idx >= num_folds):
            raise ValueError("idx must be in range 0-{}, inclusive. Received {}".format(num_folds - 1, idx))
        train = []
        for i in range(num_folds):
            if i == idx:
                vali = self._folds[i]
            else:
                train += self._folds[i]
                
        train_sequences, train_labels = zip(*train)
        vali_sequences, vali_labels = zip(*vali)
        train_fold = Fold(train_sequences, train_labels)
        vali_fold = Fold(vali_sequences, vali_labels)

        return train_fold, vali_fold

class CharCnnDataset(data.Dataset):
    def __init__(self, file_path, vocab=None, transform=None):
        self.file_path = file_path
        self.transform = transform
        self._vocab = Vocabulary() if vocab is None else vocab
        self.sequences = []
        self.labels = []
        self.max_length = self._get_max_length_and_vocab(file_path)
        self.alphabet_size = len(self._vocab)
        self._read_data()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        sequence = sequence if self.transform is None else self.transform(sequence)
        label = self.labels[idx]
        return sequence, label

    def get_vocab(self):
        return self._vocab

    def _get_max_length_and_vocab(self, file_path):
        max_length = 0
        with open(self.file_path, 'r', encoding='utf-8') as f:
            label_line = True
            for line in f:
                line = line.strip().lower()
                if label_line:
                    label_line = False
                else:
                    max_length = max(max_length, len(line))
                    seq = list(line)
                    for token in seq:
                        self._vocab.add(token)
                    label_line = True
        return max_length

    def _read_data(self):
        """Read a file in FASTA format. Specifically, resembles:
                >0
                ATCG
            where the first line is assumed to be a label line.
            Will read from self._file_path and store sequences in 
            self.sequences and labels in self.labels.
        """
        with open(self.file_path, 'r', encoding='utf-8') as f:
            label_line = True
            for line in f:
                line = line.strip().lower()
                if label_line:
                    split = line.split('>')
                    assert len(split) == 2
                    label = int(split[1])
                    assert label in [-1, 0, 1]
                    label = torch.tensor([label], dtype=torch.long)
                    self.labels.append(label)
                    label_line = False
                else:
                    seq = list(line)
                    length = len(seq)
                    # maxlen x alphabet_size
                    t = torch.zeros(self.max_length, self.alphabet_size)
                    seq = [self._vocab.add(token) for token in seq]
                    seq = torch.tensor(seq, dtype=torch.long)
                    t[:length,:] = F.one_hot(seq, num_classes=self.alphabet_size)
                    self.sequences.append(t)
                    label_line = True

        assert len(self.sequences) == len(self.labels)
        