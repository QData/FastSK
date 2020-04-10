#ifndef FASTSK_SVM_H
#define FASTSK_SVM_H

#include "fastsk_kernel.hpp"
#include "shared.h"
#include "libsvm-code/svm.h"
#include "libsvm-code/eval.h"
#include <string>
#include <cstdlib>
#include <vector>

using namespace std;

class FastSK_SVM {
public:
    int g;
    int m;
    int k;
    int num_threads = -1;
    int svm_type = C_SVC;
    int kernel_type = LINEAR;       // must be LINEAR, fastsk, or RBF
    std::string kernel_type_name;   
    double C;                       //C param
    double nu;                      //nu for nu-SVC
    double cache_size = 100;        // cache size
    double eps;                     //epsilon, tolernace of termination criteria
    int h = 1;                      //0 or 1, whether to use the shrinking heuristics
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
    double* K;
    int nfeat;

    FastSK_SVM(int, int, double, double, double, const string, bool, double*, int, int, int*, int);
    void fit();
    void predict(string predictions_file);
    double score(string metric);
    double cv(vector<vector<int> > X, vector<int> Y, int num_folds);
    svm_problem* create_svm_problem(double *, int*, svm_parameter*);
    svm_model* train_model(double*, int*, svm_parameter*);
};

#endif
