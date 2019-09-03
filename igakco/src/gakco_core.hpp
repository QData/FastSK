#ifndef GAKCO_CORE_H
#define GAKCO_CORE_H

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
	pthread_mutex_t *mutexes;
	int num_mutex;
	WorkItem *workQueue;
	int queueSize;
	bool quiet;

} kernel_params;

typedef struct train_params {
	
} train_params;

void do_work(int tid, kernel_params *params);
void do_work2(int tid);
double* construct_kernel(kernel_params *params);
void kernel_build_parallel(int tid, WorkItem *workQueue, int queueSize,
	pthread_mutex_t *mutexes, kernel_params *params, double *Ksfinal);
svm_model *train_model(double *K, int *labels, 
	kernel_params *kernel_param, svm_parameter *svm_param);
double *construct_test_kernel(int n_str_train, int n_str_test, double *K);
double *run_cross_validation(double *K, std::string metric, int k);
svm_problem *create_svm_problem(double *K, int *labels, 
	kernel_params *kernel_param, svm_parameter *svm_param);

#endif