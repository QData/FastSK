#ifndef FASTSK_H
#define FASTSK_H

#include "shared.h"
#include "libsvm-code/libsvm.h"
#include <thread>

typedef struct kernel_params {
    int g;
    int k;
    int m;
    
    long int n_str_train;
    long int n_str_test;
    long int total_str;
    long int n_str_pairs;
    Feature *features;
    int dict_size;

    int num_threads;
    int num_mutex;
    WorkItem *workQueue;
    int queueSize;
    bool quiet;
    bool approx;
    double delta;
    int max_iters;
    bool skip_variance;

} kernel_params;

class KernelFunction {
    kernel_params* params;

public:
    std::vector<double> stdevs;
    KernelFunction(kernel_params*);
    double* compute_kernel();
    void kernel_build_parallel(int, WorkItem*, int, pthread_mutex_t*, kernel_params*, double*);
    double get_variance(unsigned int*, double*, double *, int, int, int);
};

svm_model *train_model(double *K, int *labels, 
    kernel_params *kernel_param, svm_parameter *svm_param);
double *construct_test_kernel(int n_str_train, int n_str_test, double *K);
double *run_cross_validation(double *K, std::string metric, int k);
svm_problem *create_svm_problem(double *K, int *labels, 
    kernel_params *kernel_param, svm_parameter *svm_param);

#endif
