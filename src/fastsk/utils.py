"""Utils for reading fasta files
"""

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
