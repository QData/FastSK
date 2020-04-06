#ifndef KERNEL_H
#define KERNEL_H

#include <vector>
#include <string>

class Kernel {

public:
    int g;
    int m;
    int k;
    int num_threads = -1;
    int num_mutex = -1;
    int n_str_train;
    int n_str_test;
    double* kernel = NULL;
    bool quiet = false;
    bool approx = false;
    double delta = 0.025;
    int max_iters = -1;
    bool skip_variance = false;
    std::vector<double> stdevs;

    Kernel(int, int, int, bool, double, int, bool);
    void compute(std::vector<std::vector<int> >, 
        std::vector<std::vector<int> >);
    void compute_train(std::vector<std::vector<int> > Xtrain);
    std::vector<std::vector<double> > train_kernel();
    std::vector<std::vector<double> > test_kernel();
    std::vector<double> get_stdevs();
    void save_kernel(std::string);
};

#endif
