#ifndef DATASET_H
#define DATASET_H

#include "shared.h"

#include <string>
#include <vector>

class Dataset {
public:
	std::string filename;
	std::string dictFileName;
	std::vector<std::string> sequences;
	bool is_test_set;
	char *dict;
	int dictionarySize;
	int *seqLabels = (int *) malloc(MAXNSTR * sizeof(int));
	int *seqLengths = (int *) malloc(MAXNSTR * sizeof(int));
	long int nStr = 0;
	long int maxlen = 0;
	long int minlen = STRMAXLEN;
	int **S;

	Dataset(std::string filename, bool is_test_set, char *dict, std::string dictFileName);
	void get_sequences();
	void collect_data2();
	void collect_data(bool quiet);
	void readDict();
	void parseDict();
	void free_strings();
	void get_sequences_and_labels();
};

class ArrayDataset {
public:
	std::vector<std::string> sequences;
	std::vector<int> labels;
	std::vector<int> seqLengths;
	char *dict;
	int dictionarySize;
	long int n_str = 0;
	long int maxlen = 0;
	long int minlen = STRMAXLEN;
	int **S;

	ArrayDataset(std::vector<std::string> sequences, std::vector<int> labels);
	void read_data();
	void numericize_seqs();
};

class TestArrayDataset : public ArrayDataset {
public:
	TestArrayDataset(std::vector<std::string> sequences, std::vector<int> labels, char *dict);
};

#endif
