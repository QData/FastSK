import os.path as osp
import subprocess
dna_datasets = [
    'CTCF', 'EP300', 'JUND', 'RAD21', 'SIN3A',
    'Pbde', 'EP300_47848', 'KAT2B', 'TP53', 'ZZZ3',
    'Mcf7', 'Hek29', 'NR2C2', 'ZBTB33'
]

prot_datasets = [
    '1.1', '1.34', '2.1', '2.19', '2.31', '2.34',
    '2.41', '2.8', '3.19', '3.25', '3.33', '3.50'
]

nlp_datasets = [
    'AImed', 'BioInfer', 'CC1-LLL', 'CC2-IEPA', 
    'CC3-HPRD50', 'DrugBank', 'MedLine'
]

datasets = dna_datasets + prot_datasets + nlp_datasets

for dataset in datasets:
    '''s = input("Confirm to train on " + dataset + "\n")
    if s != "y":
        continue'''
    train_file = osp.join('../../data/', dataset + '.train.fasta')
    test_file = osp.join('../../data/', dataset + '.test.fasta')
    log_dir = 'log/{}_cnn_results'.format(dataset)
    epochs = 200

    command = ['python', 'run_cnn.py', '--trn', train_file,
    '--tst', test_file, '--log_dir', log_dir, '--epochs', str(epochs)]
    print(' '.join(command))
    output = subprocess.check_output(command)
    #print("done with " + dataset)
