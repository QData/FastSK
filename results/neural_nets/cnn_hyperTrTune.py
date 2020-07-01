from datetime import datetime, date
import os
import os.path as osp
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
from utils import AverageMeter, FastaReader
from models import SeqLSTM, CharacterLevelCNN

#def hyper(opt_method, lr, trn_size, train_file, test_file, datasetTag):
def get_args():
    parser = argparse.ArgumentParser(description='Sequence char-CNN Baselines')
    parser.add_argument('-b', '--batch', type=int, default=64, metavar='N',
        help='input batch size for training (default: 64)')
    parser.add_argument('--trn', type=str, help='Training file', default='./testdata/ZZZ3.train.fasta')
    parser.add_argument('--tst', type=str, help='Test file', default='./testdata/ZZZ3.test.fasta')
    parser.add_argument('--file', type=str, required=False, help='File to grid search results to')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA')
    parser.add_argument('--num-folds', type=int, default=5, help='Number of folds for CV')
    parser.add_argument('--epochs', type=int, default=20, help='Maximum number of epochs')
    parser.add_argument('--log_dir', type=str, default='./results', 
        help='Directory for storing logs, results, and checkpoints')
    parser.add_argument('--trn_size', type=float, choices=[0.2, 0.4, 0.6, 0.8, 1.], default=1)
    parser.add_argument('--opt_mtd', type=str, choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('--lr', type=float, choices=[1e-2, 3e-2, 8e-3], default=1e-2)
    parser.add_argument('--datasetTag', type=str, default='ZZZ3', help='which data')
    return parser.parse_args()

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

# Old utility func
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

# Old utility func
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

    return 


# Old utility func
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

    return 


def train_cnn(model, training_generator, optimizer, criterion, epoch, print_every=25):
    model.train()
    
    losses = AverageMeter()
    accuracies = AverageMeter()

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
            logits.cpu().detach().numpy(), metrics_list=['accuracy'])

        acc = training_metrics['accuracy']
        losses.update(loss.data, samples.size(0))
        accuracies.update(acc, samples.size(0))

        y_true += labels.cpu().numpy().tolist()
        y_pred += torch.max(logits, 1)[1].cpu().numpy().tolist()
        pos_scores += logits.cpu().detach().numpy()[:,1].tolist()

        lr = optimizer.state_dict()["param_groups"][0]["lr"]

        if (iter % print_every == 0) and (iter > 0):
            print_str = "[Training - Epoch: {}], LR: {}, Iteration: {}/{}, Loss: {}, Accuracy: {}"
            print(print_str.format(
                epoch + 1,
                lr,
                iter,
                num_iter_per_epoch,
                losses.avg,
                accuracies.avg,
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

    progress_bar = tqdm(enumerate(validation_generator),
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
            logits.cpu().detach().numpy(), metrics_list=['accuracy'])
        acc = validation_metrics['accuracy']
        losses.update(loss.data, samples.size(0))
        accuracies.update(acc, samples.size(0))

        y_true += labels.cpu().numpy().tolist()
        y_pred += torch.max(logits, 1)[1].cpu().numpy().tolist()
        pos_scores += logits.cpu().detach().numpy()[:,1].tolist()

        if (iter % print_every == 0) and (iter > 0):
            print_str = "[Validation - Epoch: {}], Iteration: {}/{}, Loss: {}, Accuracy: {}"
            print(print_str.format(
                epoch + 1,
                iter,
                num_iter_per_epoch,
                losses.avg,
                accuracies.avg,
            ))

    try:
        vali_auc = metrics.roc_auc_score(y_true, pos_scores)
    except ValueError as e:
        vali_auc = 0
        exit()

    print("Avg loss: {}, Acc: {}, AUC: {}".format(losses.avg.item(), accuracies.avg.item(), vali_auc))
    return losses.avg.item(), accuracies.avg.item(), vali_auc


# Utility func
def char_cnn_cv(args, k=5):
    ## Read the samples, get labels, max length, and alphabet size, 
    fasta = FastaReader(train_file, test_file)
    fasta.get_data()
    alphabet_size, max_len = fasta.alphabet_size, fasta.max_len
    samples, labels = fasta.train_samples, fasta.train_labels
    test_samples, test_labels = fasta.test_samples, fasta.test_labels
    
    num_samples = len(samples)
    fold_size = num_samples // k
    for i in range(k):
        ## Validation fold indices
        start = i * fold_size
        end = start + fold_size if i < k - 1 else num_samples
        vali_samples = samples[start:end]
        train_samples = samples[0:start] + samples[end:]

        ## Create PyTorch Datasets for the CharCNN
        trainset = CharCnnDataset(train_samples, train_labels, max_len, alphabet_size)
        testset = CharCnnDataset(test_samples, test_labels, max_len, alphabet_size)

        ## Initialize Model
    return



def run_main():
    ## Read the samples, get labels, max length, and alphabet size, 
    fasta = FastaReader(train_file, test_file)
    fasta.get_data()
    alphabet_size, max_len = fasta.alphabet_size, fasta.max_len

    ## Train and a random validation sets (20% of training samples)
    num_train, num_test = int(args.trn_size*fasta.num_train), fasta.num_test
    num_vali = int(0.2 * num_train)
    num_train_split = num_train - num_vali
    train_samples, train_labels = fasta.train_samples[:num_train_split], fasta.train_labels[:num_train_split]
    vali_samples, vali_labels = fasta.train_samples[num_train_split:num_train], fasta.train_labels[num_train_split:num_train]
    
    ## Test set
    test_samples, test_labels = fasta.test_samples, fasta.test_labels

    ## Create PyTorch Datasets
    trainset = CharCnnDataset(train_samples, train_labels, max_len, alphabet_size)
    valiset = CharCnnDataset(vali_samples, vali_labels, max_len, alphabet_size)
    testset = CharCnnDataset(test_samples, test_labels, max_len, alphabet_size)

    ## Create Data Loaders
    training_params = {
        "batch_size": args.batch,
        "shuffle": True,
        "drop_last": True
    }

    validation_params = {
        "batch_size": args.batch,
        "shuffle": False,
        "drop_last": True
    }

    training_generator = data.DataLoader(trainset, **training_params)
    validation_generator = data.DataLoader(valiset, **validation_params)
    test_generator = data.DataLoader(testset, **validation_params)

    ## Initialize model
    cnn_args = {
        'max_length': max_len,
        'number_of_characters': alphabet_size,
        'dropout_input': 0.1,
        'batch_size': args.batch
    }

    model = CharacterLevelCNN(cnn_args, number_of_classes=2)
    model = model.cuda() if use_cuda else model
    criterion = nn.CrossEntropyLoss()

    best_auc, best_epoch = 0, 0
    if args.opt_mtd == 'sgd': 
        opt = optim.SGD(model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=0.00001)
    elif args.opt_mtd == 'adam':
        opt = optim.Adam(model.parameters(),
            lr=1e-3,
            weight_decay=0.00001)

    ## Train
    for epoch in range(args.epochs): 
        train_loss, train_acc, train_auc = train_cnn(model, training_generator,
            optimizer=opt,
            criterion=criterion,
            epoch=epoch)

        vali_loss, vali_acc, vali_auc = evaluate_cnn(model, validation_generator,
            criterion=criterion,
            epoch=epoch)

        print_str = "[Epoch: {} / {}]\ttrain_loss: {:.4f} \ttrain_acc: {:.4f} \ttrain_auc: {:.4f}"
        print_str += " \tval_loss: {:.4f} \tval_acc: {:.4f} \tval_auc: {:.4f}"
        print(print_str.format(epoch + 1, args.epochs, train_loss, train_acc, train_auc, vali_loss, vali_acc, vali_auc))
        print("=" * 50)

        # model checkpoint
        if vali_auc > best_auc:
            best_auc = vali_auc
            best_epoch = epoch
            model_name = 'model_{}_lr_{}_trnsize_{}.pth'
            model_name = model_name.format("charcnn", 
                opt.state_dict()['param_groups'][0]['lr'],
                args.trn_size)

            torch.save(model.state_dict(), osp.join(log_dir, model_name))

    print("Best AUC = {}, Best Epoch = {}".format(best_auc, best_epoch))

    ## Retrain and evaluate on test set
    train_and_vali = CharCnnDataset(train_samples + vali_samples, 
        train_labels + vali_labels, max_len, alphabet_size)
    train_and_vali_generator = data.DataLoader(trainset, **training_params)

    model = CharacterLevelCNN(cnn_args, number_of_classes=2)
    model = model.cuda() if use_cuda else model
    criterion = nn.CrossEntropyLoss()

    if args.opt_mtd == 'sgd': 
        opt = optim.SGD(model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=0.00001)
    elif args.opt_mtd == 'adam':
        opt = optim.Adam(model.parameters(),
            lr=1e-3,
            weight_decay=0.00001)


    for epoch in range(best_epoch):
        train_loss, train_acc, train_auc = train_cnn(model, train_and_vali_generator,
            optimizer=opt,
            criterion=criterion,
            epoch=epoch)
        print_str = "[Epoch: {} / {}]\ttrain_loss: {:.4f} \ttrain_acc: {:.4f} \ttrain_auc: {:.4f}"
        print(print_str.format(epoch + 1, args.epochs, train_loss, train_acc, train_auc))
        print("=" * 50)

    test_loss, test_acc, test_auc = evaluate_cnn(model, test_generator,
        criterion=criterion,
        epoch=best_epoch)

    summary_str = "Best epoch: {}, Final test acc: {}, Final test auc: {}"
    summary_str = summary_str.format(best_epoch, test_acc, test_auc)

    with open(output_file, 'a+') as f:
        f.write(summary_str + '\n')
    

    return test_acc, test_auc


if __name__ == '__main__':
    #args = get_args(opt_method, lr, trn_size, train_file, test_file)
    args = get_args()
    bsz = args.batch
    train_file = args.trn
    test_file = args.tst
    trn_size = args.trn_size
    num_folds = args.num_folds
    log_dir = args.log_dir
    output_file = args.file
    epochs = args.epochs
    datasetTag = args.datasetTag

    print("train_file = ", train_file)
    print("test_file = ", test_file)
    print("trn_size = ", trn_size)
    print("opt_method = ", args.opt_mtd)
    print("lr = ", args.lr)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print("device = ", device)

    highest_auc = 0
    best_params = {}
    if not osp.exists(log_dir):
        os.makedirs(log_dir)

    if args.file is None:
        
        # output_file = "charcnn_results_{}.out".format(str(date.today()))
        output_file = "charcnn_results.out"
        output_file = osp.join(log_dir, output_file)

    if os.path.exists(output_file):

        with open(output_file, 'a+') as f:
            f.write("{}\nopt:{}, lr:{}, trn: {}, tst: {}, trn_size:{},  batch: {}, out: {}\n".format(datetime.now(),
                args.opt_mtd, args.lr, train_file, test_file, args.trn_size, bsz, output_file))
    else:
        with open(output_file, 'w+') as f:
            f.write("{}\nopt:{}, lr:{}, trn: {}, tst: {}, trn_size:{},  batch: {}, out: {}\n".format(datetime.now(),
                args.opt_mtd, args.lr, train_file, test_file, args.trn_size, bsz, output_file))

    test_acc_list = []
    test_auc_list = []
    for i in range(5):
        test_acc, test_auc = run_main()
        test_acc_list.append(test_acc)
        test_auc_list.append(test_auc)
    mean_acc = np.mean(np.array(test_acc_list))
    mean_auc = np.mean(np.array(test_auc_list))
    var_acc = np.var(np.array(test_acc_list))
    var_auc = np.var(np.array(test_auc_list))
    sum_str = "mean_acc: {}, var_acc: {}, mean_auc: {}, var_auc: {}"
    sum_str = sum_str.format(mean_acc, var_acc, mean_auc, var_auc)
    with open(output_file, 'a+') as f:
        f.write(sum_str + '\n')      