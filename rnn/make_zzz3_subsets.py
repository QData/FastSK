import random

file = 'data/ZZZ3.train.fasta'
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

subset_sizes = []
for i in range(1, 11):
	size = i * 1000
	if size > num_sample:
		size = num_sample
	zipped = list(zip(labels, sequences))
	random.shuffle(zipped)
	labels_, sequences_ = zip(*zipped)
	labels_, sequences_ = labels_[:size], sequences_[:size]

	assert len(labels_) == len(sequences_)
	assert len(labels_) <= num_sample

	with open('data/ZZZ3_{}.train.fasta'.format(i), 'w+') as f:
		for (label, seq) in zip(labels_, sequences_):
			f.write(label + '\n' + seq + '\n')
