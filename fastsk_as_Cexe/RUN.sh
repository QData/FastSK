#!/bin/bash

if [ $# \> 0 ]; then
	train_file=$1
	test_file=$2
	dict_file=$3
else										# Change if desired:
	train_file="./data/1.1.train.fasta"		# default training file
	test_file="./data/1.1.test.fasta"		# default testing file
	dict_file="./data/protein.dictionary.txt"		# default dictionary file
fi

echo "Training fasta file: $train_file"
echo "Testing fasta file: $test_file"
echo "Dictionary file: $dict_file"

ntrain=`wc -l $train_file | awk '{print $1/2}'`

ntest=`wc -l $test_file | awk '{print $1/2}'`

mkdir -p results

echo "Combining training and testing files to creates ./results/sequences.fasta..."

cat $train_file $test_file > ./results/sequences.fasta

echo "Feeding data into GaKCo..."

# feed into GaKCo 
#GaKCo -g <int> -k <int> -n <int> -p <int> <sequencefile> <dictionaryfile> <labelsFile> <kernelFile>
g=7 #user's choice
k=5 #user's choice
p=1	#enable or disable multithreading
n=15000 #increase if dataset has more than 15000 strings
./bin/GaKCo -g $g -k $k -n $n -p $p ./results/sequences.fasta $dict_file ./results/labels.txt ./results/kernel.txt

echo "Kernel matrix computed and stored in results/kernel.txt..."

echo "Using kernel to create kernel_train.txt and kernel_test.txt..."

# cut the kernel into train and test
cat ./results/kernel.txt | cut -d' ' -f1-$ntrain | head -$ntrain > ./results/kernel_train.txt
cat ./results/kernel.txt | cut -d' ' -f1-$ntrain | tail -n -$ntest > ./results/kernel_test.txt
#cut labels
cat ./results/labels.txt | cut -d' ' -f1 | head -$ntrain > ./results/train.labels.txt
cat ./results/labels.txt | cut -d' ' -f1 | tail -n $ntest > ./results/test.labels.txt

echo "Creating empirical feature map for svmlight format..."

#create empirical feature map (liblinear/libsvm/svmlight format : label followed by features)
paste -d" " ./results/train.labels.txt ./results/kernel_train.txt>./results/train.features.txt
paste -d" " ./results/test.labels.txt ./results/kernel_test.txt>./results/test.features.txt

echo "Finished"

#Feeding results into svmlight requires installation of svmlight and placement of 
#svm_learn executable in the GaKCo-SVM/bin directory
#C parameter for liblinear/libsvm/svmlight
#C=1
#q parameter
#Q=10 #this is the svmlight default
#use liblinear/libsvm/svmlight train function (user's choice)
#./bin/svm_learn -c $C -q $Q ./results/train.features.txt ./results/model.txt