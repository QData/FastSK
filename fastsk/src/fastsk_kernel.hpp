#ifndef FASTSK_KERNEL_H
#define FASTSK_KERNEL_H

#include "shared.h"
#include "libsvm-code/svm.h"
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

svm_model* train_model(double*, int*, kernel_params*, svm_parameter*);
double* construct_test_kernel(int, int, double*);

#endif
