#ifndef KERNEL_H
#define KERNEL_H

#include <vector>
#include <string>

using namespace std;

class Kernel {

public:
    int g;
    int m;
    int k;
    int num_threads = -1;
    int num_mutex = -1;
    int n_str_train;
    int n_str_test;
    vector<vector<int> > Xtrain;
    vector<vector<int> > Xtest;
    vector<int> test_labels;
    double* kernel = NULL;
    bool quiet = false;
    bool approx = false;
    double delta = 0.025;
    int max_iters = -1;
    bool skip_variance = false;
    vector<double> stdevs;
    int nfeat;

    Kernel(int, int, int, bool, double, int, bool);
    void compute_kernel(const string, const string, const string);
    void compute_kernel(const string, const string);
    void compute_kernel(vector<string>, vector<string>);
    void compute_kernel(vector<vector<int> >, vector<vector<int> >);
    void compute_train(vector<vector<int> > Xtrain);
    void fit(double, double, double, const string);
    vector<vector<double> > train_kernel();
    vector<vector<double> > test_kernel();
    vector<double> get_stdevs();
    void save_kernel(string);
};

#endif
