#include "kernel.hpp"
#include "gakco_core.hpp"
#include "shared.h"
#include <vector>
#include <array>
#include <string>
#include <set>
#include <math.h>
#include <cstring>

Kernel::Kernel(int g, int m) {
    this->g = g;
    this->m = m;
    this->k = g - m;
}

void Kernel::compute(std::vector<std::vector<int> > Xtrain, 
    std::vector<std::vector<int> > Xtest) {

    std::vector<int> lengths;
    for (int i = 0; i < Xtrain.size(); i++) {
        lengths.push_back(Xtrain[i].size());
    }
    for (int i = 0; i < Xtest.size(); i++) {
        lengths.push_back(Xtest[i].size());
    }
    
    long int n_str_train = Xtrain.size();
    long int n_str_test = Xtest.size();
    long int total_str = n_str_train + n_str_test;

    this->n_str_train = n_str_train;
    this->n_str_test = n_str_test;

    int **S = (int **) malloc(total_str * sizeof(int*));
    int *seq_lengths = (int *) malloc(total_str * sizeof(int));
    int *labels = (int *) malloc(total_str * sizeof(int));
    int *test_labels = (int *) malloc(n_str_test * sizeof(int));

    std::memcpy(seq_lengths, lengths.data(), lengths.size() * sizeof(int));

    std::set<int> dict;
    dict.insert(0);
    for (int i = 0; i < n_str_train; i++) {
        S[i] = Xtrain[i].data();
        for (int j = 0; j < seq_lengths[i]; j++) {
            dict.insert(Xtrain[i][j]);
        }
    }
    int dictionarySize = dict.size();
    printf("dictionarySize = %d\n", dictionarySize);
    for (int i = 0; i < n_str_test; i++) {
        S[n_str_train + i] = Xtest[i].data();
    }
    
    /*Extract g-mers*/
    Features* features = extractFeatures(S, seq_lengths, total_str, g);
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
