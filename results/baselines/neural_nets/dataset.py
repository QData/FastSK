### dataset.py
### A class for containing datasets used by RNN baseline models
### Derrick Blakely, December 2018

import numpy as np
import torch
import torch.nn as nn
import random
import string
from torchnlp.word_to_vector import GloVe

CACHE = '.word_vectors_cache'

def glove_embedding(size):
    glove = GloVe('6B', size, cache=CACHE)
    stoi = {tok: i for i, tok in enumerate(glove.itos)}
    rows, cols = glove.vectors.shape
    embedding = nn.Embedding(rows, cols, _weight=glove.vectors)
    return embedding, stoi, glove.itos

class Vocabulary(object):
    def __init__(self):
        self.token2idx = {}
        self.idx2token = {}
        self.size = len(self.token2idx)
        self.pretrained = False

    @classmethod
    def from_glove(cls, size):
        self = cls()
        embedding, stoi, itos = glove_embedding(size)
        self.word2idx = stoi
        self.idx2word = dict(enumerate(itos))
        self.vocab_size = len(self.word2idx)
        self.pretrained = True
        return self, embedding

    def add(self, token):
        """Params:
        token: a letter (for char-level model) or word (for word-level model)
        for which to create a mapping to an integer (the idx).

        Return: the index of the word. If it's already present, return its
        index. Otherwise, add it before returning the index.
        """
        if token not in self.token2idx:
            self.token2idx[token] = self.size
            self.token2idx[self.size] = token
            self.size += 1
        return self.token2idx.get(token)

    def __len__(self):
        return self.size

class Dataset(object):
    def __init__(self, train_file, test_file, dictionary_file=None, 
        use_cuda=False, word_model=False, vocab=None):
        
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.word_model = word_model

        # use pre-trained glove embeddings
        if vocab:
            self.vocab = vocab
        else:
            self.vocab = Vocabulary()

        self.xtrain, self.ytrain = self.prepare_data_for_embedding(train_file)
        self.xtest, self.ytest = self.prepare_data_for_embedding(test_file)
        self.n_train = len(self.xtrain)
        self.n_test = len(self.xtest)

    def get_dict(self, dict_file):
        dictionary = {}
        num = 0
        with open(dict_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().lower()
                assert len(line) == 1
                assert line not in dictionary
                dictionary[line] = num
                num += 1
        return dictionary

    def prepare_data_for_embedding(self, datafile):
        """Represent sequences as tensors of char indexes (no one-hot)
        Intended to be used in conjunction with torch.nn.Embedding
        """
        numerical_seqs = []
        labels = []
        nlp_exclude = set(string.punctuation)
        with open(datafile, 'r', encoding='utf-8') as f:
            label_line = True # first line is assumed to be a label line
            for line in f:
                line = line.strip().lower()
                if label_line:
                    split = line.split('>')
                    assert len(split) == 2
                    label = int(split[1])
                    assert label == 0 or label == 1
                    label_tensor = torch.tensor([label], dtype=torch.long, device=self.device)
                    labels.append(label_tensor)
                    label_line = False
                else:
                    if self.word_model:
                        line = ''.join(c for c in line if c not in nlp_exclude)
                        seq = line.split()
                    else:
                        seq = list(line) # character-level model
                    seq = [self.vocab.add(token) for token in seq]
                    seq = torch.tensor(seq, dtype=torch.long, device=self.device)
                    numerical_seqs.append(seq)
                    label_line = True
        assert len(numerical_seqs) == len(labels)

        return numerical_seqs, labels

    def get_batch(self, batch_size, training_data=True):
        """Retrieve a random batch of the given batch_size
        """
        xbatch = []
        ybatch = []
        max_idx = self.n_train - 1 if training_data else self.n_test - 1
        for _ in range(batch_size):
            rand = random.randint(0, max_idx)
            xbatch.append(self.xtrain[rand] if training_data else self.xtest[rand])
            ybatch.append(self.ytrain[rand] if training_data else self.ytest[rand])
        return xbatch, ybatch
        