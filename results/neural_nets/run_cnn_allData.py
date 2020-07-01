import os.path as osp
import subprocess

dna_datasets = [
    'CTCF', 'EP300', 'JUND', 'RAD21', 'SIN3A',
    'Pbde', 'EP300_47848', 'KAT2B', 'TP53', 'ZZZ3',
]

prot_datasets = [
    '1.1', '1.34', '2.19', '2.31', '2.34',
    '2.41', '2.8', '3.19', '3.25', '3.33',
]

nlp_datasets = [
    'AIMed', 'BioInfer', 'CC1-LLL', 'CC2-IEPA', 
    'CC3-HPRD50', 'DrugBank', 'MedLine'
]

datasets = prot_datasets + dna_datasets + nlp_datasets

for dataset in datasets:
    train_file = osp.join('../../data/', dataset + '.train.fasta')
    test_file = osp.join('../../data/', dataset + '.test.fasta')
    log_dir = '{}_cnn_results'.format(dataset)
    epochs = 200

    command = ['python', 'run_cnn.py', '--trn', train_file,
    '--tst', test_file, '--log_dir', log_dir, '--epochs', str(epochs)]
    print(' '.join(command))
    output = subprocess.check_output(command)
