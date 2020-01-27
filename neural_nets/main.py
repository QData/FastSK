### main.py

import numpy as np
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn import metrics
from tqdm import tqdm, trange
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import Vocabulary, FastaDataset, CharCnnDataset, collate, get_evaluation
from utils import AverageMeter
from models import SeqLSTM, CharacterLevelCNN

def get_args():
    parser = argparse.ArgumentParser(description='Bio-Sequence RNN Baselines')
    parser.add_argument('-b', '--batch', type=int, default=64, metavar='N',
        help='input batch size for training (default: 64)')
    parser.add_argument('--trn', type=str, required=True, help='Training file', metavar='1.1.train.fasta')
    parser.add_argument('--tst', type=str, required=True, help='Test file', metavar='1.1.test.fasta')
    parser.add_argument('--file', type=str, required=True, help='File to grid search results to')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA')
    parser.add_argument('--num-folds', type=int, default=5, help='Number of folds for CV')
    parser.add_argument('--epochs', type=int, default=20, help='Maximum number of epochs')

    return parser.parse_args()

args = get_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print("device = ", device)
bsz = args.batch
train_file = args.trn
test_file = args.tst
epochs = args.epochs
output_file = args.file
highest_auc = 0
best_params = {}
num_folds = args.num_folds

with open(output_file, 'w+') as f:
    f.write("trn: {} tst: {}, batch: {}, out: {}".format(train_file,
        test_file, bsz, output_file))

def train_epoch(model, opt, train_loader):
    num_batches = len(train_loader)
    epoch_loss = 0
    for x, y, lengths in train_loader:
        opt.zero_grad()
        x, y = x.to(device), y.to(device)
        logits = model(x, lengths)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        epoch_loss += loss.item()
    
    return epoch_loss / num_batches

def evaluate(model, test_loader):
    num_batches = len(test_loader)
    num_samples = 0
    epoch_loss = 0
    num_correct = 0
    true_ys = []
    preds = []
    scores = []
    with torch.no_grad():
        for x, y, lengths in test_loader:
            num_samples += y.shape[0]
            x, y = x.to(device), y.to(device)
            logits = model(x, lengths)
            loss = F.cross_entropy(logits, y)
            epoch_loss += loss.item()
            y_pred = logits.max(dim=1)[1]
            num_correct += (y_pred == y).sum().item()
            probs = F.softmax(logits, dim=1)
            pos_scores = probs[:,1].tolist()
            scores += pos_scores
            pos_score = probs[0][1].item()
            true_ys += y.tolist()
            preds += y_pred.tolist()

    epoch_loss /= num_batches
    accuracy = (num_correct / num_samples) * 100
    confusion = metrics.confusion_matrix(true_ys, preds)
    #print(str(confusion) + '\n')
    # true positive rate/sensitvity
    tpr = 100 * confusion[1][1] / (confusion[1][0] + confusion[1][1])
    # true negative rate/specificity
    tnr = 100 * confusion[0][0] / (confusion[0][0] + confusion[0][1])
    # AUROC
    try:
        auc = metrics.roc_auc_score(true_ys, scores)
    except ValueError as e:
        with open(output_file, 'a+') as f:
            f.write(str(e) + "\n")
            f.write("true_ys = {}\n, scores = {}".format(true_ys, scores))
        auc = 0
    
    return epoch_loss, accuracy, tpr, tnr, auc

def run(params, trainset):
    global highest_auc, best_params
    print(params)

    trainset.split()
    total_acc = 0
    total_auc = 0

    num_epochs = 0
    for i in range(num_folds):
        train, vali = trainset.get_fold(i)
        train_loader = data.DataLoader(train, 
            batch_size=bsz, 
            shuffle=False,
            collate_fn=collate)
        vali_loader = data.DataLoader(vali,
            batch_size=bsz,
            shuffle=False,
            collate_fn=collate)
        model = SeqLSTM(device=device, input_size=params['input_size'],
            embedding_size=params['embedding_size'],
            hidden_size=params['hidden_size'],
            output_size=params['output_size'],
            n_layers=params['n_layers'],
            bidir=params['bidir'],
            embedding=None).to(device)
        opt = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=0.0005)

        for i in range(1, epochs + 1):
            num_epochs = i
            train_loss = train_epoch(model, opt, train_loader)
            vali_loss, acc, tpr, tnr, auc = evaluate(model, vali_loader)
            # early stopping criterion
            if vali_loss > train_loss:
                break

        vali_loss, acc, tpr, tnr, auc = evaluate(model, vali_loader)
        result = "train_loss = {}, vali_loss = {}, ".format(train_loss, vali_loss)
        result += "acc = {}, tpr/sensitvity = {}, ".format(acc, tpr)
        result += "tnr/specificity = {}, AUROC = {}".format(tnr, auc)
        total_acc += acc
        total_auc += auc

    acc = total_acc / num_folds
    auc = total_auc / num_folds

    if (auc > highest_auc):
        highest_auc = auc
        best_params = params
        best_params['num_epochs'] = num_epochs

    with open(output_file, 'a+') as f:
        f.write("\n\n" + str(params) + 'num_epochs: ' + str(num_epochs) + '\n' + result + '\n')

def run_best(trainset, testset):
    train_loader = data.DataLoader(trainset, 
        batch_size=bsz, 
        shuffle=False,
        collate_fn=collate)
    test_loader = data.DataLoader(testset,
        batch_size=bsz,
        shuffle=True,
        collate_fn=collate)
    model = SeqLSTM(device=device, input_size=best_params['input_size'],
        embedding_size=best_params['embedding_size'],
        hidden_size=best_params['hidden_size'],
        output_size=best_params['output_size'],
        n_layers=best_params['n_layers'],
        bidir=best_params['bidir'],
        embedding=None).to(device)
    opt = optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=0.0005)

    for i in range(1, best_params['num_epochs'] + 1):
        train_epoch(model, opt, train_loader)

    test_loss, test_acc, test_tpr, test_tnr, test_auc = evaluate(model, test_loader)
    result = "loss = {}, acc = {}, ".format(test_loss, test_acc)
    result += "tpr/sensitvity = {}, tnr/specificity = {}, ".format(test_tpr, test_tnr)
    result += "AUROC = {}".format(test_auc)

    with open(output_file, 'a+') as f:
        f.write("\n\nFinal model: " + str(best_params) + '\n' + result + '\n')

#def train_cnn(model, training_generator, optimizer, criterion, epoch, writer, log_file, scheduler, class_names, args, print_every=25):
def train_cnn(model, training_generator, optimizer, criterion, epoch, print_every=25):
    model.train()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    aucs = AverageMeter()

    num_iter_per_epoch = len(training_generator)

    y_true, y_pred, pos_scores = [], [], []

    progress_bar = tqdm(enumerate(training_generator),
        total=num_iter_per_epoch)

    for iter, batch in progress_bar:
        samples, labels = batch
        labels = labels.squeeze(1)

        if use_cuda:
            samples = samples.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        logits = model(samples)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        ## training metrics
        training_metrics = get_evaluation(labels.cpu().numpy(), 
            logits.cpu().detach().numpy())
        acc, auc = training_metrics['accuracy'], training_metrics['auc']
        losses.update(loss.data, samples.size(0))
        accuracies.update(acc, samples.size(0))
        aucs.update(auc, samples.size(0))

        y_true += labels.cpu().numpy().tolist()
        y_pred += torch.max(logits, 1)[1].cpu().numpy().tolist()
        pos_scores += logits.cpu().detach().numpy()[:,1].tolist()

        lr = optimizer.state_dict()["param_groups"][0]["lr"]

        if (iter % print_every == 0) and (iter > 0):
            print_str = "[Training - Epoch: {}], LR: {}, Iteration: {}/{}, Loss: {}, Accuracy: {}, AUC: {}"
            print(print_str.format(
                epoch + 1,
                lr,
                iter,
                num_iter_per_epoch,
                losses.avg,
                accuracies.avg,
                aucs.avg
            ))

    try:
        train_auc = metrics.roc_auc_score(y_true, pos_scores)
    except ValueError as e:
        print("y_prob = ", y_prob)
        print("y_true = ", y_true)
        train_auc = 0
        exit()

    print("Avg loss: {}, Acc: {}, AUC: {}".format(losses.avg.item(), accuracies.avg.item(), train_auc))
    return losses.avg.item(), accuracies.avg.item(), train_auc

def evaluate_cnn(model, validation_generator, criterion, epoch, print_every=25):
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    aucs = AverageMeter()
    num_iter_per_epoch = len(validation_generator)

    y_true, y_pred, pos_scores = [], [], []

    progress_bar = tqdm(enumerate(training_generator),
        total=num_iter_per_epoch)

    for iter, batch in progress_bar:
        samples, labels = batch
        labels = labels.squeeze(1)
        if use_cuda:
            samples = samples.cuda()
            labels = labels.cuda()

        with torch.no_grad():
            logits = model(samples)

        loss = criterion(logits, labels)

        ## validation metrics
        validation_metrics = get_evaluation(labels.cpu().numpy(),
            logits.cpu().detach().numpy())
        acc, auc = validation_metrics['accuracy'], training_metrics['auc']
        losses.update(loss.data, samples.size(0))
        accuracies.update(acc, samples.size(0))
        aucs.update(auc, samples.size(0))

        y_true += labels.cpu().numpy().tolist()
        y_pred += torch.max(logits, 1)[1].cpu().numpy().tolist()
        pos_scores += logits.cpu().detach().numpy()[:,1].tolist()

        if (iter % print_every == 0) and (iter > 0):
            print_str = "[Validation - Epoch: {}], Iteration: {}/{}, Loss: {}, Accuracy: {}, AUC: {}"
            print(print_str.format(
                epoch + 1,
                iter,
                num_iter_per_epoch,
                losses.avg,
                accuracies.avg,
                aucs.avg
            ))

    try:
        vali_auc = metrics.roc_auc_score(y_true, pos_scores)
    except ValueError as e:
        vali_auc = 0
        exit()

    print("Avg loss: {}, Acc: {}, AUC: {}".format(losses.avg.item(), accuracies.avg.item(), vali_auc))
    return losses.avg.item(), accuracies.avg.item(), vali_auc


def run_char_cnn(args):
    pass

def main():
    trainset = CharCnnDataset(train_file)
    alphabet = trainset.get_vocab()
    testset = CharCnnDataset(test_file, alphabet) 
    
    max_len = trainset.max_length
    
    cnn_args = {
        'max_length': max(trainset.max_length, testset.max_length), # not dynamic/per batch
        'number_of_characters': len(alphabet), # alphabet size
        'dropout_input': 0.1, # default value from repo
        'batch_size': 64 # temp - read from console
    }

    training_params = {
        "batch_size": 64,
        "shuffle": True,
        "drop_last": True
    }

    validation_params = {
        "batch_size": 64,
        "shuffle": False,
        "drop_last": True
    }

    training_generator = data.DataLoader(trainset, **training_params)
    test_generator = data.DataLoader(testset, **validation_params)

    model = CharacterLevelCNN(cnn_args, number_of_classes=2)
    model = model.cuda() if use_cuda else model
    criterion = nn.CrossEntropyLoss()

    opt = optim.SGD(model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=0.00001)

    for epoch in range(args.epochs): 
        train_cnn(model, training_generator,
            optimizer=opt,
            criterion=criterion,
            epoch=epoch)

if __name__ == '__main__':
    main()
    