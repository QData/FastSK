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

	Kernel(int, int, int);
	void compute(std::vector<std::vector<int> >, 
		std::vector<std::vector<int> >);
	void compute_train(std::vector<std::vector<int> > Xtrain);
	std::vector<std::vector<double> > train_kernel();
	std::vector<std::vector<double> > test_kernel();
	void save_kernel(std::string);
};

#endif
