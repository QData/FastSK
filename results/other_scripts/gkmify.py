import os
from os import path as osp
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Convert our data to gkm format')
    parser.add_argument('--dir', type=str, required=False, default='./',
        help='Directory where data in our format is stored')
    parser.add_argument('--prefix', type=str, required=True, 
        help='Dataset prefix name', metavar='EP300')
    parser.add_argument('--out_dir', type=str, required=False, default='gkm_format')

    return parser.parse_args()

uniqueID = 0

def read_and_convert(train_file):
    global uniqueID
    pos_data = []
    neg_data = []
    with open(train_file, 'r', encoding='utf-8') as f:
        label_line = True
        pos = True
        for line in f:
            line = line.strip().lower()
            if label_line:
                uniqueID += 1
                split = line.split('>')
                assert len(split) == 2
                label = int(split[1])
                assert label in [-1, 0, 1]
                if label == 1:
                    pos = True
                else:
                    pos = False
                gkm_label = '>' + str(uniqueID)
                if label == 1:
                    pos_data.append(gkm_label)
                else:
                    neg_data.append(gkm_label)
                label_line = False
            else:
                if pos:
                    pos_data.append(line)
                else:
                    neg_data.append(line)
                label_line = True
    return pos_data, neg_data

def write_data(data, file):
    with open(file, 'w+') as f:
        f.writelines('\n'.join(data))
        f.write('\n')

# Read args
args = get_args()
dir = args.dir
out_dir = args.out_dir
if not osp.exists(out_dir):
    os.makedirs(out_dir)
prefix = args.prefix
train_file = osp.join(args.dir, prefix + ".train.fasta")
test_file = osp.join(args.dir, prefix + ".test.fasta")

# Get read data in our format, convert it into a 
pos_train_data, neg_train_data = read_and_convert(train_file)
pos_test_data, neg_test_data = read_and_convert(test_file)

# create the gkm style data files
pos_train_name = osp.join(out_dir, prefix + '.train.pos.fasta')
neg_train_name = osp.join(out_dir, prefix + '.train.neg.fasta')
pos_test_name = osp.join(out_dir, prefix + '.test.pos.fasta')
neg_test_name = osp.join(out_dir, prefix + '.test.neg.fasta')

write_data(pos_train_data, pos_train_name)
write_data(neg_train_data, neg_train_name)
write_data(pos_test_data, pos_test_name)
write_data(neg_test_data, neg_test_name)
