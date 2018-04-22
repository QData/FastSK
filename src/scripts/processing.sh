#!/bin/bash

#train/test/dictionary files
train_file="./data/CTCF.train.fasta"
test_file="./data/CTCF.test.fasta"
dict_file="./data/dict-dna.txt"

ntrain=`wc -l $train_file | awk '{print $1/2}'`

ntest=`wc -l $test_file | awk '{print $1/2}'`
cat $train_file $test_file > sequences.fasta

echo $ntrain
echo $ntest
# feed into GaKCo 
#GaKCo <sequencefile> <dictionaryfile> <filename  for labels> <g> <k> <filename for kernel> <set for multithread>   
#g,k (user's choice)
g=7
k=5
./GaKCo sequences.fasta ./data/CTCF.test.fasta $dict_file labels.txt $g $k kernel.txt 1
#./GaKCo sequences.fasta ./data/dict-protein.txt labels.txt 7 5 kernel.txt 1
#./GaKCo ./data/1.1.train.fasta ./data/1.1.test.fasta ./data/dict-protein.txt labels.txt 7 5 kernel.txt 1 .01
#./GaKCo ./data/CTCF.train.fasta ./data/CTCF.test.fasta ./data/dict-dna.txt labels.txt 7 5 kernel.txt 1 1

# cut the kernel into train and test
cat kernel.txt | cut -d' ' -f1-$ntrain | head -$ntrain > kernel_train.txt
cat kernel.txt | cut -d' ' -f1-$ntrain | tail -n -$ntest > kernel_test.txt
#cut labels
cat labels.txt | cut -d' ' -f1 | head -$ntrain > train.labels.txt
cat labels.txt | cut -d' ' -f1 | tail -n $ntest > test.labels.txt


#create empirical feature map (liblinear/libsvm/svmlight format : label foloowed by features)
paste -d" " train.labels.txt kernel_train.txt>train.features.txt
paste -d" " test.labels.txt kernel_test.txt>test.features.txt
 #C parameter for liblinear/libsvm/svmlight
 C=1
#use liblinear/libsvm/svmlight train function (user's choice)
train -c $C -q train.features.txt model.txt
