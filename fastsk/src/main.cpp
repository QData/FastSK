#include "fastsk.hpp"
#include <string>

#include <stdlib.h>

int main() {
	// kernel parameters
	int g = 8;
	int m = 4;
	int t = 20;
	bool approx = false;
	double delta = 0.025;
	int max_iters = 100;
	bool skip_variance = false;

	const string xtrain = "../../data/1.1.train.fasta";
	const string xtest = "../../data/1.1.test.fasta";
	FastSK* fastsk = new FastSK(g, m, t, approx, delta, max_iters, skip_variance);	
	fastsk->compute_kernel(xtrain, xtest);
	
	// svm parameters
	double C = 0.01;
	double nu = 1;
	double eps = 1;
	const string kernel_type = "linear";

	fastsk->fit(C, nu, eps, kernel_type);

}