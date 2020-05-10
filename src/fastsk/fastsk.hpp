#ifndef FASTSK_H
#define FASTSK_H

#include <vector>
#include <string>
#include "fastsk_kernel.hpp"
#include "libsvm-code/svm.h"
#include "libsvm-code/eval.h"

using namespace std;



class FastSK {
    int g;
    int m;
    int k;
    int num_threads = -1;
    int num_mutex = -1;
    int svm_type = C_SVC;
    int kernel_type = LINEAR;       // must be LINEAR, FASTSK, or RBF
    string kernel_type_name;   
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
    double* test_kernel = NULL;
    long int total_str;
    long int n_str_train;
    long int n_str_test;
    int numClasses = -1;
    char *dictionary;
    bool quiet = false;
    svm_model *model;
    int nfeat;
    vector<vector<int> > Xtrain;
    vector<vector<int> > Xtest;
    int* train_labels;
    int* test_labels;
    double* K = NULL;
    bool approx = false;
    double delta = 0.025;
    int max_iters = -1;
    bool skip_variance = false;
    vector<double> stdevs;

public:
    FastSK(int, int, int, bool, double, int, bool);
    void compute_kernel(const string, const string, const string);
    void compute_kernel(const string, const string);
    void compute_kernel(vector<string>, vector<string>);
    void compute_kernel(vector<vector<int> >, vector<vector<int> >);
    void compute_train(vector<vector<int> > Xtrain);
    vector<vector<double> > get_train_kernel();
    vector<vector<double> > get_test_kernel();
    vector<double> get_stdevs();
    void save_kernel(string);
    void fit(double, double, double, const string);
    svm_model* train_model(double *, int *, svm_parameter *);
    svm_problem* create_svm_problem(double *, int *, svm_parameter *);
    double score(const string);
};

#endif
