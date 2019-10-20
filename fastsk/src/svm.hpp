#ifndef GAKCOSVM_H
#define GAKCOSVM_H

#include "shared.h"
#include "libsvm-code/libsvm.h"
#include <string>
#include <cstdlib>
#include <vector>

class SVM {
public:
	int g;
	int m;
	int k;
	int num_threads = -1;
	int num_mutex = -1;
	int svm_type = C_SVC;
	int kernel_type = LINEAR; // must be LINEAR, GAKCO, or RBF
	std::string kernel_type_name;
	double C; //C param
	double nu; //nu for nu-SVC
	double cache_size = 100; // cache size
	double eps; //epsilon, tolernace of termination criteria
	int h = 1; //0 or 1, whether to use the shrinking heuristics
	int nr_weight = 0;
	int *weight_label = NULL;
	double *weight = NULL;
	int probability = 1;
	int crossfold = 0;
	double* kernel = NULL;
	double* test_kernel = NULL;
	long int total_str;
	long int n_str_train;
	long int n_str_test;
	int *test_labels;
	int numClasses = -1;
	char *dictionary;
	bool quiet = false;
	svm_model *model;

	SVM(int g, int m, double C, double nu, double eps, std::string kernel);
	void toString();
	void fit_numerical(std::vector<std::vector<int> > Xtrain, 
		std::vector<int> Ytrain, std::vector<std::vector<int> > Xtest,
		std::vector<int> Ytest, std::string kernel_file);
	void fit(std::string train_file, std::string test_file, 
		std::string dict, bool quiet, std::string kernel_file);
	void predict(std::string predictions_file);
	void fit_from_arrays(std::vector<std::string> Xtrain, std::vector<int> Ytrain,
		std::vector<std::string> Xtest, std::vector<int> Ytest, 
		std::string kernel_file);
	double score(std::string metric);
	double cv(std::vector<std::vector<int> > X, std::vector<int> Y, int num_folds);
};

#endif