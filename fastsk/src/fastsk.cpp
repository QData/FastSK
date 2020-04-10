#include "fastsk.hpp"
#include "fastsk_kernel.hpp"
#include "shared.h"
#include "utils.hpp"
#include "fastsk_svm.hpp"

#include <vector>
#include <string>
#include <set>
#include <math.h>
#include <cstring>
#include <iostream>

using namespace std;

FastSK::FastSK(int g, int m, int t, bool approx, double delta, int max_iters, bool skip_variance) {
    this->g = g;
    this->m = m;
    this->k = g - m;
    this->num_threads = t;
    this->approx = approx;
    this->delta = delta;
    this->max_iters = max_iters;
    this->skip_variance = skip_variance;
}

void FastSK::compute_kernel(const string Xtrain, const string Xtest) {
    const string dictionary_file = "";
    this->compute_kernel(Xtrain, Xtest, dictionary_file);
}

void FastSK::compute_kernel(const string Xtrain, const string Xtest, const string dictionary_file) {
    // Read in the sequences from the two files and convert them vectors
    DataReader* data_reader = new DataReader(Xtrain, dictionary_file);
    bool train = true;

    data_reader->read_data(Xtrain, train);
    data_reader->read_data(Xtest, !train);
    vector<vector<int> > train_seq = data_reader->train_seq;
    vector<int> train_labels = data_reader->train_labels;
    vector<vector<int> > test_seq = data_reader->test_seq;
    vector<int> test_labels = data_reader->test_labels;

    this->test_labels = test_labels;

    this->compute_kernel(train_seq, test_seq);

}

void FastSK::compute_kernel(vector<string> Xtrain, vector<string> Xtest) {
    // Convert sequences to numerical form (as vectors)

}

void FastSK::compute_kernel(vector<vector<int> > Xtrain, vector<vector<int> > Xtest) {
    // Given sequences already in numerical form, compute the kernel matrix
    vector<int> lengths;
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

    cout << "Length of shortest train sequence: " << shortest_train << endl;
    cout << "Length of shortest test sequence: " << shortest_test << endl;

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

    set<int> dict;
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
    int dict_size = dict.size();
    cout << "Dictionary size = " << dict_size << " (+1 for unknown char)." << endl;
    
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
    params.dict_size = dict_size;
    params.num_threads = this->num_threads;
    params.num_mutex = this->num_mutex;
    params.quiet = this->quiet;
    params.approx = this->approx;
    params.delta = this->delta;
    params.max_iters = this->max_iters;
    params.skip_variance = this->skip_variance;

    KernelFunction* kernel_function = new KernelFunction(&params);
    double *K = kernel_function->compute_kernel();

    this->kernel = K;
    this->stdevs = kernel_function->stdevs;
}

void FastSK::fit(double C, double nu, double eps, const string kernel_type) {
    cout << "Creating SVM with params: " << endl;
    cout << "\tC = " << C << endl;
    cout << "\tnu = " << nu << endl;
    cout << "\teps = " << eps << endl;
    cout << "\tkernel_type = " << kernel_type << endl;

    //SVM* svm = new FastSK_SVM(this->g, this->m, C, nu, eps, kernel_type);

    int g = this->g;
    int m = this->m;
    bool quiet = false;
    int n_str_train = this->n_str_train;
    int n_str_test = this->n_str_test;
    int* test_labels = this->test_labels.data();
    int nfeat = this->nfeat;

    FastSK_SVM* svm = new FastSK_SVM(g, 
        m,
        C, 
        nu, 
        eps, 
        kernel_type, 
        quiet, 
        this->kernel, 
        n_str_train,
        n_str_test, 
        test_labels,
        nfeat
    );

    svm->fit();
}

void FastSK::compute_train(vector<vector<int> > Xtrain) {
    vector<int> lengths;
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

    set<int> dict;
    dict.insert(0);
    for (int i = 0; i < n_str_train; i++) {
        S[i] = Xtrain[i].data();
        for (int j = 0; j < lengths[i]; j++) {
            dict.insert(Xtrain[i][j]);
        }
    }

    int dict_size = dict.size();
    cout << "Dictionary size = " << dict_size << " (+1 for unknown char)." << endl;
    
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
    params.dict_size = dict_size;
    params.num_threads = this->num_threads;
    params.num_mutex = this->num_mutex;
    params.quiet = this->quiet;
    params.approx = this->approx;
    params.delta = this->delta;
    params.max_iters = this->max_iters;
    params.skip_variance = this->skip_variance;

    KernelFunction* kernel_function = new KernelFunction(&params);
    double *K = kernel_function->compute_kernel();

    this->kernel = K;
    this->stdevs = kernel_function->stdevs;
    this->nfeat = nfeat;
}

vector<vector<double> > FastSK::train_kernel() {
    double *K = this->kernel;
    int n_str_rain = this->n_str_train;
    vector<vector<double> > train_K(n_str_train, vector<double>(n_str_train, 0));
    for (int i = 0; i < n_str_train; i++) {
        for (int j = 0; j < n_str_train; j++) {
            train_K[i][j] = tri_access(K, i, j);
        }
    }
    return train_K;
}

vector<vector<double> > FastSK::test_kernel() {
    double *K = this->kernel;
    int n_str_train = this->n_str_train;
    int n_str_test = this->n_str_test;
    int total_str = this->n_str_train + this->n_str_test;

    vector<vector<double> > test_K(n_str_test, vector<double>(n_str_train, 0));

    for (int i = n_str_train; i < total_str; i++){
        for (int j = 0; j < n_str_train; j++){
            test_K[i - n_str_train][j]  = tri_access(K, i, j);
        }
    }

    return test_K;
}

vector<double> FastSK::get_stdevs() {
    return this->stdevs;
}

void FastSK::save_kernel(string kernel_file) {
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
