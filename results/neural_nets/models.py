import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

"""Character or word-level LSTM
"""


class SeqLSTM(nn.Module):
    # input_size = alphabet_size
    def __init__(
        self,
        device,
        input_size,
        embedding_size,
        hidden_size,
        output_size,
        n_layers=1,
        bidir=False,
        embedding=None,
    ):

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
            self.embedding = nn.Embedding(
                num_embeddings=input_size, embedding_dim=embedding_size
            )

        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            bidirectional=bidir,
        )
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
        h0 = torch.zeros(
            self.n_layers * self.num_dir, batch, self.hidden_size, device=self.device
        )
        c0 = torch.zeros(
            self.n_layers * self.num_dir, batch, self.hidden_size, device=self.device
        )
        return h0, c0


"""Character-level CNN
Based on "Character-level Convolutional Networks for Text Classification"
    https://arxiv.org/pdf/1509.01626.pdf
Code adapted from
    https://github.com/ahmedbesbes/character-based-cnn
"""


class CharacterLevelCNN(nn.Module):
    def __init__(self, args, number_of_classes=2):
        super(CharacterLevelCNN, self).__init__()

        self.dropout_input = nn.Dropout2d(args["dropout_input"])

        # define conv layers
        self.conv1 = nn.Sequential(
            nn.Conv1d(args["number_of_characters"], 256, kernel_size=7, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(3),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=7, padding=0), nn.ReLU(), nn.MaxPool1d(3)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=0), nn.ReLU()
        )

        ### kernel sizes need to be adjusted with DNA, else
        ## "kernel size can't be greater than actual input size"

        # self.conv4 = nn.Sequential(
        #     nn.Conv1d(256, 256, kernel_size=3, padding=0),
        #     nn.ReLU()
        # )

        # self.conv5 = nn.Sequential(
        #     nn.Conv1d(256, 256, kernel_size=3, padding=0),
        #     nn.ReLU()
        # )

        # self.conv6 = nn.Sequential(
        #     nn.Conv1d(256, 256, kernel_size=3, padding=0),
        #     nn.ReLU(),
        #     nn.MaxPool1d(3)
        # )

        # compute the  output shape after forwarding an input to the conv layers
        input_shape = (
            args["batch_size"],
            args["max_length"],
            args["number_of_characters"],
        )
        self.output_dimension = self._get_conv_output(input_shape)

        # define linear layers
        self.fc1 = nn.Sequential(
            nn.Linear(self.output_dimension, 1024), nn.ReLU(), nn.Dropout(0.5)
        )

        self.fc2 = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.5))

        self.fc3 = nn.Linear(1024, number_of_classes)

        # initialize weights
        self._create_weights()

    def forward(self, x):
        x = self.dropout_input(x)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)
        # x = self.conv6(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    # utility private functions
    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)

    def _get_conv_output(self, shape):
        x = torch.rand(shape)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)
        # x = self.conv6(x)
        x = x.view(x.size(0), -1)
        output_dimension = x.size(1)
        return output_dimension
