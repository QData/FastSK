#ifndef GAKCO_CORE_H
#define GAKCO_CORE_H

#include "libsvm-code/libsvm.h"
#include <thread>

#define STRMAXLEN 15000
#define MAXNSTR 15000

typedef struct Feature {
    int *features;
    int *group;
    int n;
    ~Feature() {
        free(features);
        free(group);
    }
} Features;

typedef struct Combinations {
    int n;
    int k;
    double num_comb;
    int *comb;
} Combinations;

typedef struct Dict {
    int index;
    char *word;
} Dict;

typedef struct WorkItem {
    int m;
    int combo_num;
} WorkItem;

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
    bool approx;
    double epsilon;

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

Features* extractFeatures(int **S, std::vector<int> seqLengths, int nStr, int g);
Features* extractFeatures(int **S, int* seqLengths, int nStr, int g);
double& tri_access(double* array, int i, int j);
unsigned int& tri_access(unsigned int* array, int i, int j, int N);
unsigned int& tri_access(unsigned int* array, int i, int j);
char *trimwhitespace(char *s);
std::string trim(std::string& s);
void cntsrtna(unsigned int *out,unsigned int *sx, int k, int r, int na);
void countAndUpdate(unsigned int *outK, unsigned int *sx, unsigned int *g, int k, int r, int nStr);
void countAndUpdateTri(unsigned int *outK, unsigned int *sx, unsigned int *g, int k, int r, int nStr);
unsigned nchoosek(unsigned n, unsigned k);
void getCombinations(unsigned int n, unsigned int k, int *pos, unsigned int depth, unsigned int margin, unsigned int *cnt_comb, unsigned int *out, int num_comb);
void shuffle(WorkItem *array, size_t n);
void print_null(const char *s);
void validate_args(int g, int m);
void g_greater_than_shortest_err(int g, int len, std::string filename);
void g_greater_than_shortest_train(int g, int len);
void g_greater_than_shortest_test(int g, int len);
double calculate_auc(double* pos, double* neg, int npos, int nneg);

#endif
