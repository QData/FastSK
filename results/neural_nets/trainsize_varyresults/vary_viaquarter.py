import random
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
total = dna_datasets + prot_datasets + nlp_datasets

for s in total:
    file = '../../data/' + s + '.train.fasta'
    labels, sequences = [], []

    # load the data
    with open(file, 'r') as f:
        label_line = True
        for line in f:
            line = line.strip()
            if label_line:
                labels.append(line)
                label_line = False
            else:
                sequences.append(line)
                label_line = True
    assert len(labels) == len(sequences)
    num_sample = len(labels)
    zipped = list(zip(labels, sequences))
    random.shuffle(zipped)
    zipped = zipped[int(len(zipped)/4)*(4-1):] # takes 1/4 of the data
    with open('quarter_data/' + set + '.train.fasta', 'w+') as f:
        for (label, seq) in zipped:
            f.write(label + '\n' + seq + '\n')
