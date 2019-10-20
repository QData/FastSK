#include "kernel.hpp"
#include "gakco_core.hpp"
#include "shared.h"
#include <vector>
#include <array>
#include <string>
#include <set>
#include <math.h>
#include <cstring>

Kernel::Kernel(int g, int m, int t, bool approx, double epsilon, int max_iters) {
    this->g = g;
    this->m = m;
    this->k = g - m;
    this->num_threads = t;
    this->approx = approx;
    this->epsilon = epsilon;
    this->max_iters = max_iters;
}

void Kernel::compute(std::vector<std::vector<int> > Xtrain, 
    std::vector<std::vector<int> > Xtest) {

    std::vector<int> lengths;
    int shortest_train = Xtrain[0].size();
    for (int i = 0; i < Xtrain.size(); i++) {
        int len = Xtrain[i].size();
        if (len < shortest_train) {
            shortest_train = len;
        }
        lengths.push_back(len);
    }
    int shortest_test = Xtest[0].size();
    for (int i = 0; i < Xtest.size(); i++) {
        int len = Xtest[i].size();
        if (len < shortest_test) {
            shortest_test = len;
        }
        lengths.push_back(len);
    }
    printf("shortest train sequence: %d, shortest test sequence: %d\n", shortest_train, shortest_test);
    if (this->g > shortest_train) {
        g_greater_than_shortest_train(this->g, shortest_train);
    }
    if (this->g > shortest_test) {
        g_greater_than_shortest_test(this->g, shortest_test);
    }
    
    long int n_str_train = Xtrain.size();
    long int n_str_test = Xtest.size();
    long int total_str = n_str_train + n_str_test;

    this->n_str_train = n_str_train;
    this->n_str_test = n_str_test;

    int **S = (int **) malloc(total_str * sizeof(int*));

    std::set<int> dict;
    dict.insert(0);
    for (int i = 0; i < n_str_train; i++) {
        S[i] = Xtrain[i].data();
        for (int j = 0; j < lengths[i]; j++) {
            dict.insert(Xtrain[i][j]);
        }
    }
    for (int i = 0; i < n_str_test; i++) {
        S[n_str_train + i] = Xtest[i].data();
        for (int j = 0; j < lengths[n_str_train + i]; j++) {
            dict.insert(Xtest[i][j]);
        }
    }
    int dictionarySize = dict.size();
    for (int d : dict) {
        printf("%d,", d);
    }
    printf("\n");
    printf("dictionarySize = %d\n", dictionarySize);
    
    /*Extract g-mers*/
    Features* features = extractFeatures(S, lengths, total_str, g);
    int nfeat = (*features).n;
    int *feat = (*features).features;
    if (!this->quiet) {
        printf("g = %d, k = %d, %d features\n", this->g, this->k, nfeat);
    }

    kernel_params params;
    params.g = g;
    params.k = k;
    params.m = m;
    params.n_str_train = n_str_train;
    params.n_str_test = n_str_test;
    params.total_str = total_str;
    params.n_str_pairs = (total_str / (double) 2) * (total_str + 1);
    params.features = features;
    params.dict_size = dictionarySize;
    params.num_threads = this->num_threads;
    params.num_mutex = this->num_mutex;
    params.quiet = this->quiet;
    params.approx = this->approx;
    params.epsilon = this->epsilon;
    params.max_iters = this->max_iters;

    /* Compute the kernel matrix */
    double *K = construct_kernel(&params);
    this->kernel = K;
}

void Kernel::compute_train(std::vector<std::vector<int> > Xtrain) {

    std::vector<int> lengths;
    int shortest_train = Xtrain[0].size();
    for (int i = 0; i < Xtrain.size(); i++) {
        int len = Xtrain[i].size();
        if (len < shortest_train) {
            shortest_train = len;
        }
        lengths.push_back(len);
    }

    if (this->g > shortest_train) {
        g_greater_than_shortest_train(this->g, shortest_train);
    }
    
    long int n_str_train = Xtrain.size();
    long int n_str_test = 0;
    long int total_str = n_str_train + n_str_test;

    this->n_str_train = n_str_train;
    this->n_str_test = n_str_test;

    int **S = (int **) malloc(total_str * sizeof(int*));

    std::set<int> dict;
    dict.insert(0);
    for (int i = 0; i < n_str_train; i++) {
        S[i] = Xtrain[i].data();
        for (int j = 0; j < lengths[i]; j++) {
            dict.insert(Xtrain[i][j]);
        }
    }

    int dictionarySize = dict.size();
    for (int d : dict) {
        printf("%d,", d);
    }
    printf("\n");
    printf("dictionarySize = %d\n", dictionarySize);
    
    /*Extract g-mers*/
    Features* features = extractFeatures(S, lengths, total_str, g);
    int nfeat = (*features).n;
    int *feat = (*features).features;
    if (!this->quiet) {
        printf("g = %d, k = %d, %d features\n", this->g, this->k, nfeat);
    }

    kernel_params params;
    params.g = g;
    params.k = k;
    params.m = m;
    params.n_str_train = n_str_train;
    params.n_str_test = n_str_test;
    params.total_str = total_str;
    params.n_str_pairs = (total_str / (double) 2) * (total_str + 1);
    params.features = features;
    params.dict_size = dictionarySize;
    params.num_threads = this->num_threads;
    params.num_mutex = this->num_mutex;
    params.quiet = this->quiet;
    params.approx = this->approx;
    params.epsilon = this->epsilon;
    params.max_iters = this->max_iters;

    /* Compute the kernel matrix */
    double *K = construct_kernel(&params);
    this->kernel = K;
}

std::vector<std::vector<double> > Kernel::train_kernel() {
    double *K = this->kernel;
    int n_str_rain = this->n_str_train;
    std::vector<std::vector<double> > train_K(n_str_train, std::vector<double>(n_str_train, 0));
    for (int i = 0; i < n_str_train; i++) {
        for (int j = 0; j < n_str_train; j++) {
            train_K[i][j] = tri_access(K, i, j);
        }
    }
    return train_K;
}

std::vector<std::vector<double> > Kernel::test_kernel() {
    double *K = this->kernel;
    int n_str_train = this->n_str_train;
    int n_str_test = this->n_str_test;
    int total_str = this->n_str_train + this->n_str_test;

    std::vector<std::vector<double> > test_K(n_str_test, std::vector<double>(n_str_train, 0));

    for (int i = n_str_train; i < total_str; i++){
        for (int j = 0; j < n_str_train; j++){
            test_K[i - n_str_train][j]  = tri_access(K, i, j);
        }
    }

    return test_K;
}

void Kernel::save_kernel(std::string kernel_file) {
    double *K = this->kernel;
    int total_str = this->n_str_train + this->n_str_test;
    if (!kernel_file.empty()) {
        printf("Writing kernel to %s...\n", kernel_file.c_str());
        FILE *kernelfile = fopen(kernel_file.c_str(), "w");
        for (int i = 0; i < total_str; ++i) {
            for (int j = 0; j < total_str; ++j) {
                fprintf(kernelfile, "%d:%e ", j + 1, tri_access(K,i,j));
            }
            fprintf(kernelfile, "\n");
        }
        fclose(kernelfile);
    }
}
