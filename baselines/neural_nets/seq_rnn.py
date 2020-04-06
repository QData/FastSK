### seq_rnn.py
### RNN baseline models for comparisons with iGakco-SVM
### Derrick Blakely, December 2018

### General Imports
import numpy as np
import os
import sys
import shutil
import matplotlib
# A less ad hoc way of setting these backends would be nice
if os.environ['HOME'] == "/Users/derrick":
    matplotlib.use('TkAgg') # Need to use TkAgg backend for my machine
else:
    matplotlib.use('Agg') # Need to use Agg backend for the qdata nodes
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import random
import argparse
from dataset import Dataset, Vocabulary
#from utils import FastaDataset, Vocabulary
from tqdm import tqdm, trange
from sklearn import metrics
import datetime
import seaborn as sn

### Torch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data

def get_args():
    parser = argparse.ArgumentParser(description='Bio-Sequence RNN Baselines')
    parser.add_argument('-b', '--batch', type=int, default=64, metavar='N',
        help='input batch size for training (default: 64)')
    parser.add_argument('-i', '--iters', type=int, required=True, metavar='N',
        help='number of iterations to train (default: 1000)')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.01, metavar='LR',
        help='learning rate (default: 0.01)')
    parser.add_argument('-em', '--embed-size', type=int, default=32,
        help='Size of the embedding space (using char-level embeddings')
    parser.add_argument('--layers', type=int, default=1, metavar='N',
        help='Number of RNN layers to stack')
    parser.add_argument('--bidir', action='store_true', default=False,
        help='Whether to use a bidirectional RNN')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
        help='SGD momentum (default: 0.5)')
    parser.add_argument('--hidden', type=int, default=64, metavar='N',
        help='Number of hidden units (default: 64)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
        help='disables CUDA')
    parser.add_argument('-li', '--log-interval', type=int, default=1000, metavar='N',
        help='how many iterations to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
        help='For Saving the current Model')
    parser.add_argument('-opt', '--opt', choices=['adagrad', 'adam', 'sgd'], default='sgd',
        help='Which optimizers to use. Options are Adagrad, Adam, SGD')
    parser.add_argument('--trn', type=str, required=True, help='Training file', metavar='1.1.train.fasta')
    parser.add_argument('--tst', type=str, required=True, help='Test file', metavar='1.1.test.fasta')
    parser.add_argument('--show-graphs', action='store_true', default=False,
        help='Will show plots of the training and test accuracy and the training loss over time')
    parser.add_argument('-od', '--output-directory', type=str,
        help="Name of directory to create. Will save data inside the directory."
            "If not provided, logged data will not be saved.")
    parser.add_argument('-d', '--dict', required=False, type=str, metavar='dna.dictionary',
        help='Dictionary file containing all chars that can appear in sequences, 1 per line')
    parser.add_argument('-pw', '--pos-weight', type=float, default=1,
        help='Weighting factor to place on the positive class')
    parser.add_argument('-nw', '--neg-weight', type=float, default=1,
        help='Weighting factor to place on the negative class')
    parser.add_argument('--glove', type=int, choices=[50, 100, 200, 300], required=False,
        help='Size of pretrained Glove embeddings. Overrides the --embed-size argument')
    parser.add_argument('--word', action='store_true', default=False,
        help='Flag to use word-level (for NLP) model instead of char-level model')
    
    return parser.parse_args()

args = get_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()

'''Change visible devices with:
$ CUDA_VISIBLE_DEVICES=0 python seq_rnn.py [args]
'''

device = torch.device('cuda' if use_cuda else 'cpu')
train_file = args.trn
test_file = args.tst
iters = args.iters
log_interval = args.log_interval
n_layers = args.layers
bidir = args.bidir
BATCH = args.batch
lr = args.learning_rate
PAD_VAL = -1

class History(object):
    def __init__(self):
        self.acc_iters = []
        self.train_acc = []
        self.test_acc = []
        self.loss_iters = []
        self.losses = []
        self.auc_iters = []
        self.train_auc = []
        self.test_auc = []
        self.fpr, self.tpr = [], []

    def add_acc(self, iter, train, test):
        self.acc_iters.append(iter)
        self.train_acc.append(train)
        self.test_acc.append(test)

    # training loss (e.g., NLL Loss)
    def add_loss(self, iter, loss):
        self.loss_iters.append(iter)
        self.losses.append(loss)

    # for plotting changes in auc-roc over time
    def add_auc(self, iter, train, test):
        self.auc_iters.append(iter)
        self.train_auc.append(train)
        self.test_auc.append(test)

    # data needed for plotting a single ROC
    def add_roc_info(self, fpr, tpr):
        self.fpr = fpr
        self.tpr = tpr

    def plot_acc(self, show=False, path=None):
        plt.plot(self.acc_iters, self.train_acc, label='Train Accuracy')
        plt.plot(self.acc_iters, self.test_acc, label='Test Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Training Iteration')
        plt.legend(['Train', 'Test'], loc='upper left')
        if show: plt.show()
        if path is not None:
            file_name = os.path.join(path, 'accuracy.pdf')
            plt.savefig(file_name)
            plt.clf()

    def plot_loss(self, show=False, path=None):
        plt.plot(self.loss_iters, self.losses, label='NLL Loss')
        plt.title('Model Training Loss')
        plt.ylabel('Negative Log-Likelihood Loss')
        plt.xlabel('Training Iteration')
        if show: plt.show()
        if path is not None:
            file_name = os.path.join(path, 'loss.pdf')
            plt.savefig(file_name)
            plt.clf()

    # test auc vs iters plot
    # shows changes in auc-roc over time (not an ROC curve itself)
    def plot_auc(self, show=False, path=None):
        plt.plot(self.auc_iters, self.test_auc, label='Test AUC')
        plt.title('Test Set AUC-ROC vs Training Iterations')
        plt.ylabel('AUC-ROC')
        plt.xlabel('Training Iteration')
        if show: plt.show()
        if path is not None:
            file_name = os.path.join(path, 'auc.pdf')
            plt.savefig(file_name)
            plt.clf()
    
    # an actual ROC
    def plot_roc(self, show=False, path=None):
        try:
            plt.plot(self.fpr, self.tpr, label='ROC Curve')
            plt.title('ROC Curve')
            plt.ylabel('TPR')
            plt.xlabel('FPR')
            if show: plt.show()
            if path is not None:
                file_name = os.path.join(path, 'roc.pdf')
                plt.savefig(file_name)
                plt.clf()
        except:
            print("Make sure the History add_roc_info() was called first!")

    def save_data(self, path):
        train_acc_auc = os.path.join(path, "train_acc_auc.txt")
        test_acc_auc = os.path.join(path, "test_acc_auc.txt")
        train_loss_file = os.path.join(path, "train_loss.txt")
        roc_file = os.path.join(path, 'roc.txt')
        with open(train_acc_auc, 'w+') as f:
            f.write("Iter Acc AUROC\n")
            for i, auc, acc in zip(self.acc_iters, self.train_auc, self.train_acc):
                f.write("{}\t{}\t{}\n".format(i, acc, auc))
        with open(test_acc_auc, 'w+') as f:
            f.write("Iter Acc AUROC\n")
            for i, acc, auc in zip(self.acc_iters, self.test_auc, self.test_acc):
                f.write("{} {} {}\n".format(i, auc, acc))
        with open(train_loss_file, 'w+') as f:
            for i, loss in zip(self.loss_iters, self.losses):
                f.write("{} {}\n".format(i, loss))
        with open(roc_file, 'w+') as f:
            f.write(' '.join(map(str, self.fpr)))
            f.write('\n')
            f.write(' '.join(map(str, self.tpr)))

class Evaluation(object):
    def __init__(self, model, samples, labels):
        """
        Arguments
        ---------
        model: an LSTM
        samples: array of test or train tensors
        labels: array of label tensors
        """

        num_samples = len(samples)
        assert num_samples == len(labels)
        
        num_correct = 0
        true_ys = []
        preds = []
        scores = []

        with torch.no_grad():
            for x, y in zip(samples, labels):
                x = x.unsqueeze(1)
                y = y.item()

                h0, c0 = model.init_hidden(batch=1)
                out = model(x, h0, c0)
                #score = torch.max(out).item()
                pos_score = out[0][1].item()
                scores.append(pos_score)
                y_pred = out.argmax(dim=-1).item()
                preds.append(y_pred)
                true_ys.append(y)
                if y_pred == y: num_correct += 1

        self.accuracy = (num_correct / num_samples) * 100
        self.confusion = metrics.confusion_matrix(true_ys, preds)

        self.increasing_fprs, self.increasing_tprs, thresholds = metrics.roc_curve(true_ys, scores)
        self.auc = metrics.roc_auc_score(true_ys, scores)

        # true positive rate/sensitvity
        self.tpr = 100 * self.confusion[1][1] / (self.confusion[1][0] + self.confusion[1][1])
        # true negative rate/specificity
        self.tnr = 100 * self.confusion[0][0] / (self.confusion[0][0] + self.confusion[0][1])

    def show_confusion(self):
        print(str(self.confusion) + '\n')

    def plot_confusion(self, show=False, path=None):
        dataframe = pd.DataFrame(self.confusion, index=['y = 0', 'y = 1'],
            columns=['y_pred = 0', 'y_pred = 1'])
        sn.set(font_scale=1.4)
        heatmap = sn.heatmap(dataframe, annot=True, annot_kws={'size': 16})
        fig = heatmap.get_figure()
        if show:
            plt.show()
        if path is not None:
            file_name = os.path.join(path, 'confusion.pdf')
            fig.savefig(file_name)

class SeqRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super(SeqRNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(in_features=input_size + hidden_size, 
            out_features=hidden_size)
        self.i2o = nn.Linear(in_features=input_size + hidden_size, 
            out_features=output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, hidden):
        # input ~ (batch, alphabet_size); hidden ~ (batch, hidden_size)
        # ~ (batch, alphabet_size + hidden_size)
        combined = torch.cat((input, hidden), dim=1)
        # ~ (batch, hidden_size)
        hidden = self.i2h(combined)
        # ~ (batch, output_size)
        out = self.i2o(combined)
        # ~ (batch, output_size)
        out = self.softmax(out)

        return out, hidden

    def init_hidden(self, batch):
        return torch.zeros(batch, self.hidden_size, device=device)

class SeqGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super(SeqGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(input_size, hidden_size)
        # if using embedding, should be:
        #self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, n_layers)
        
        # w/o embedding:
        self.gru = nn.GRU(input_size, hidden_size, n_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward_old(self, input, hidden):
        # input ~ (batch, alphabet)
        # hidden ~ (n_layers * directions, batch, hidden_size)
        input = input.unsqueeze(0) # make input ~ (1, batch, alphabet)

        # embedded ~ (batch, hidden_size)
        #embedded = self.embed(input)
        
        # out ~ (seqLen=1, batch, hidden)
        out, hidden = self.gru(input, hidden)
        
        # out ~ (seqLen=1, batch, output_size)
        out = self.fc(out)

        # out ~ (seqLen=1, batch, output_size) --> out ~ (batch, output_size)
        out = self.softmax(out.squeeze(0))

        return out, hidden

    def forward(self, input, hidden):
        # input ~ (max_seqlen, batch, alphabet)
        # hidden ~ (n_layers * directions, batch, hidden_size)

        # embedded ~ (batch, hidden_size)
        #embedded = self.embed(input)
        
        # out ~ (max_seqlen, batch, hidden)
        out, hidden = self.gru(input, hidden)
        #print("gru_out.size = ", out.size())
        
        # out ~ (max_seqlen, batch, output_size)
        out = self.fc(out)
        #print("fc_out.size() = ", out.size())

        # out ~ (seqLen, batch, output_size) --> out ~ (batch, output_size)
        out = self.softmax(out)
        #print("softmax_out = ", out)
        #print("softmax_out.size() = ", out.size())

        return out[-1], hidden

    def init_hidden(self, batch):
        return torch.zeros(self.n_layers, batch, self.hidden_size, device=device)

class SeqLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super(SeqLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(input_size, hidden_size)

        # w/o embedding:
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input = input.unsqueeze(0)
        out, hidden = self.lstm(input, hidden)
        out = self.fc(out)
        out = self.softmax(out.squeeze(0))
        return out, hidden

    def init_hidden(self, batch):
        hidden_state = torch.zeros(self.n_layers, batch, self.hidden_size, device=device)
        cell_state = torch.zeros(self.n_layers, batch, self.hidden_size, device=device)
        return (hidden_state, cell_state)

class BetterLSTM(nn.Module):
    # input_size = alphabet_size
    def __init__(self, input_size, embedding_size, hidden_size, output_size, 
            n_layers=1, bidir=False, embedding=None):

        super(BetterLSTM, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.num_dir = 2 if bidir else 1
        self.hidden = self.init_hidden(batch=1)

        print("input_size = ", input_size)
        print("embedding_size = ", embedding_size)

        # whether to use pre-trained embeddings
        if embedding:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_size)

        self.lstm = nn.LSTM(input_size=embedding_size, 
            hidden_size=hidden_size, num_layers=n_layers, bidirectional=bidir)
        self.fully_connected = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1) 

    def forward(self, input, h0, c0):
        '''
        embedded = self.embed(input)
        lstm_out, self.hidden = self.lstm(embedded, self.hidden)
        out = self.fully_connected(lstm_out[-1])
        scores = self.softmax(out)
        return scores
        '''
        embedded = self.embedding(input)
        output, (h_final, c_final) = self.lstm(embedded, self.hidden)
        out = self.fully_connected(h_final[-1])
        scores = self.softmax(out)

        return out

    def init_hidden(self, batch):
        h0 = torch.zeros(self.n_layers * self.num_dir, 
            batch, self.hidden_size, device=device)
        c0 = torch.zeros(self.n_layers * self.num_dir, 
            batch, self.hidden_size, device=device)
        return h0, c0

'''
def main():
    trainset = FastaDataset('./data/1.1.train.fasta')
    train_loader = data.DataLoader(trainset, batch_size=1, shuffle=True)
    alphabet = trainset.get_vocab()
    testset = FastaDataset('./data/1.1.test.fasta', alphabet)
    test_loader = data.DataLoader(testset, batch_size=1, shuffle=True)
    model = BetterLSTM(input_size=alphabet.size(), embedding_size=32,
            hidden_size=64, output_size=2,
            n_layers=2, bidir=True, embedding=None).to(device)
    loss_function = F.cross_entropy

    opt = optim.Adam(model.parameters(), lr=0.001)
    for x, y in train_loader:
        opt.zero_grad()
        x, y = x.to(device), y.to(device)
        x = torch.unsqueeze(x, dim=1) # (seqlen, sigma) --> (seqlen, batch=1, sigma)
        print("x.shape = ", x.shape)
        print("x = ", x)
        h0, c0 = model.init_hidden(batch=1)
        print("h0.shape = ", h0.shape)
        print("c0.shape = ", c0.shape)
        y_pred = model(x, h0, c0)
        loss = loss_function(y_pred, y, class_weights)
        loss.backward()
        opt.step()

if __name__ == '__main__':
    main()
'''

def main():
    parameters = {}
    embedding = vocab = None
    if args.glove is not None:
        vocab, embedding = Vocabulary.from_glove(args.glove)
        embed_size = args.glove
        word = True
    else:
        embed_size = args.embed_size
        word = args.word

    dataset = Dataset(train_file, test_file, args.dict, use_cuda, word, vocab)
    xtrain, ytrain = dataset.xtrain, dataset.ytrain
    xtest, ytest = dataset.xtest, dataset.ytest
    alphabet_size = len(dataset.vocab)
    hidden_size = args.hidden
    num_train, num_test = len(xtrain), len(xtest)
    iters = args.iters
    log_interval = args.log_interval
    epochs = iters / num_train

    print('Training size: %d ' % num_train)
    print('Test size: %d' % num_test)
    print('Alphabet size: %d' % alphabet_size)
    print('Num epochs: %s' % epochs)
    print('Device = %s' % device)

    num_classes = 2

    parameters['iterations'] = iters
    parameters['epochs'] = epochs
    parameters['log interval'] = log_interval
    parameters['algorithm'] = 'lstm'
    parameters['train set'] = train_file
    parameters['test set'] = test_file
    parameters['dictionary file'] = args.dict
    parameters['cuda'] = use_cuda
    parameters['bidir'] = bidir
    parameters['layers'] = n_layers
    parameters['embedding size'] = embed_size
    parameters['learning rate'] = lr
    parameters['hidden size'] = hidden_size
    parameters['optimizer'] = args.opt
    parameters['loss function'] = 'cross entropy'
    parameters['class weights'] = '[{}, {}]'.format(args.neg_weight, args.pos_weight)

    model = BetterLSTM(input_size=alphabet_size, embedding_size=embed_size,
        hidden_size=hidden_size, output_size=num_classes,
        n_layers=n_layers, bidir=bidir, embedding=embedding).to(device)

    if args.opt == 'adagrad':
        opt = optim.Adagrad(model.parameters(), lr=lr)
    elif args.opt == 'adam':
        opt = optim.Adam(model.parameters(), lr=lr)
    else:
        opt = optim.SGD(model.parameters(), lr=lr)

    class_weights = torch.FloatTensor([args.neg_weight, args.pos_weight]).to(device)

    loss_function = F.cross_entropy

    hist = History()
    interval_loss = 0

    for i in trange(1, iters + 1):
        opt.zero_grad()
        # sample training set
        rand = random.randint(0, num_train - 1)
        x, y = xtrain[rand], ytrain[rand]
        x = torch.unsqueeze(x, dim=1) # (seqlen, sigma) --> (seqlen, batch=1, sigma)
        h0, c0 = model.init_hidden(batch=1)
        y_pred = model(x, h0, c0)
        loss = loss_function(y_pred, y, class_weights)
        loss.backward()
        opt.step()

        interval_loss += loss.item()

        if i % log_interval == 0:
            avg_loss = interval_loss / log_interval
            hist.add_loss(i, avg_loss)
            interval_loss = 0
            
            train_eval = Evaluation(model, dataset.xtrain, dataset.ytrain)
            test_eval = Evaluation(model, dataset.xtest, dataset.ytest)

            hist.add_acc(i, train_eval.accuracy, test_eval.accuracy)
            hist.add_auc(i, train_eval.auc, test_eval.auc)
            hist.add_roc_info(test_eval.increasing_fprs, 
                test_eval.increasing_tprs)
            summary = ("Iter {}\ntrain acc = {}\ntest acc = {}\n"
                "TPR/sensitvity/recall = {}\nTNR/specificity = {}\n"
                "train loss = {}\nAUROC = {}".format(i, train_eval.accuracy, 
                    test_eval.accuracy, test_eval.tpr, test_eval.tnr,
                    avg_loss, test_eval.auc))
            print(summary)
            print("Confusion:")
            test_eval.show_confusion()

    final_train_eval = Evaluation(model, dataset.xtrain, dataset.ytrain)
    final_test_eval = Evaluation(model, dataset.xtest, dataset.ytest)

    summary = ("Final Eval:\ntrain acc = {}\ntest acc = {}\n"
        "TPR/sensitvity/recall = {}\nTNR/specificity = {}"
        "\nAUROC = {}".format(final_train_eval.accuracy, 
        final_test_eval.accuracy, final_test_eval.tpr, final_test_eval.tnr, final_test_eval.auc))
    print(summary)

    final_test_eval.show_confusion()

    parameters['accuracy test'] = final_test_eval.accuracy
    parameters['accuracy train'] = final_train_eval.accuracy
    parameters['auc test'] = final_test_eval.auc
    parameters['auc train'] = final_train_eval.auc
    parameters['TPR/sensitvity/recall'] = final_test_eval.tpr
    parameters['TNR/specificity'] = final_test_eval.tnr

    # if output_directory specified, write data for future viewing
    # otherwise, it'll be discarded
    if args.output_directory is not None:
        path = args.output_directory
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        hist.save_data(path)
        summary_file = os.path.join(path, 'about.txt')
        
        with open(summary_file, 'w+') as f:
            now = datetime.datetime.now()
            f.write(now.strftime('%Y-%m-%d %H:%M') + '\n')
            f.write('command_used: python ' + ' '.join(sys.argv) + '\n')
            f.write('(may have included CUDA_VISIBLE_DEVICES=x first)\n\n')
            for key, value in sorted(parameters.items()):
                f.write(key + ': ' + str(value) + '\n')

        hist.plot_acc(show=False, path=path)
        hist.plot_loss(show=False, path=path)
        hist.plot_auc(show=False, path=path)
        hist.plot_roc(show=False, path=path)
        final_test_eval.plot_confusion(show=False, path=path)

        print("Saved results to " + path)

    if args.show_graphs:
        hist.plot_acc(show=True)
        hist.plot_loss(show=True)
        hist.plot_auc(show=True)
        hist.plot_roc(show=True)
        final_test_eval.plot_confusion(show=True)

if __name__ == '__main__':
    main()
    