import os
import os.path as osp
import subprocess

train_sets = ['ZZZ3_{}.train.fasta'.format(i) for i in range(1, 11)]
train_sets = [osp.join('./testdata', t) for t in train_sets]
tst = osp.join('./testdata', 'ZZZ3.test.fasta')

for i in range(0, 10):
        trn, outfile = train_sets[i], 'results_{}_4.out'.format(i + 1)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "3"
        command = ['python', 'run_cnn.py',
            '-b', str(64),
            '--trn', trn, 
            '--tst', tst, 
            '--num-folds', str(5),
            '--epochs', str(20),
            '--file', outfile]
        print(' '.join(command))
        output = subprocess.check_output(command, env=env)
