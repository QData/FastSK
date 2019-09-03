import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SeqLSTM(nn.Module):
    # input_size = alphabet_size
    def __init__(self, device, input_size, embedding_size, hidden_size, output_size, 
            n_layers=1, bidir=False, embedding=None):

        super(SeqLSTM, self).__init__()

        self.device = device
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.num_dir = 2 if bidir else 1

        # whether to use pre-trained embeddings
        if embedding:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(num_embeddings=input_size, 
                embedding_dim=embedding_size)

        self.lstm = nn.LSTM(input_size=embedding_size, 
            hidden_size=hidden_size, 
            num_layers=n_layers, 
            bidirectional=bidir)
        self.fully_connected = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1) 

    def forward(self, x, lengths):
        # assumes x ~ (sequence_len, batch)
        batch_size = x.shape[1]
        hidden = self.init_hidden(batch=batch_size)
        embedded = self.embedding(x)
        packed_embedded = pack_padded_sequence(embedded, lengths)
        lstm_out, (h_final, c_final) = self.lstm(packed_embedded, hidden)
        lstm_out = pad_packed_sequence(lstm_out)
        logits = self.fully_connected(h_final[-1])

        return logits

    def init_hidden(self, batch):
        h0 = torch.zeros(self.n_layers * self.num_dir, 
            batch, self.hidden_size, device=self.device)
        c0 = torch.zeros(self.n_layers * self.num_dir, 
            batch, self.hidden_size, device=self.device)
        return h0, c0
        