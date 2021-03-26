
class FastaUtility():
    def __init__(self, vocab=None):
        r"""
        Initialize a helper object for parsing datasets in FASTA-like format.

        Parameters
        ----------
        vocab :
        """
        self._vocab = Vocabulary() if vocab is None else vocab

    def read_data(self, data_file, vocab='inferred', regression=False):
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
