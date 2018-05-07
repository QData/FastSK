//header for GakcoSVM

#ifndef GAKCOSVM_H
#define GAKCOSVM_H 
#include "libsvm-code/svm.h"

//Parameter struct, one struct created for each dataset, allows you to store different sets of parameters
//and instantiate a GackoSVM with them when you want to run them.
typedef struct gakco_param {
	std::string filename; //sequence file
	std::string testFilename; //test sequences file
	std::string dictFilename; //dictionary file
	std::string labelFilename; //label file
	std::string outputFilename; //kernel file
	std::string modelName; //name of either the model you will save or load depending on whether load model is set
	int g;
	int k;
	int threads = -1; //Number of threads by default
	int svm_type = C_SVC;
	int kernel_type = GAKCO;
	double C 		= 1.0; //C param
	double nu 		= .5; //nu for nu-SVC
	int cache_size 	= 100; //cache size
	double eps 		= .001; //epsilon, tolerance of termination criterion
	int h 			= 1; //0 or 1, whether to use the shrinking heuristics
	int nr_weight 	= 0;		/* for C_SVC */
	int *weight_label = NULL;	/* for C_SVC */
	double* weight 	= NULL;		/* for C_SVC */
	int probability = 1;
	int crossfold	= 0; //cross-fold validation mode, v-fold, 0 is no cv.
	int q 			= 0; //quiet mode
	int loadkernel = 0; //1 if this instance needs to load a precomputed kernel
	int loadmodel = 0; //1 if this instance needs to load a model
} gakco_param;

class GakcoSVM {
public:
	gakco_param *params;
	double* kernel;
	double* test_kernel;
	Features *kernel_features; //so that each test feature-set can be appended but keep the original list
	struct svm_model* model;
	int* labels;
	int* test_labels;
	long int nStr; //here instead of params because it is not set by user ever.
	long int nTestStr;

	GakcoSVM(gakco_param* arg);
	double* construct_kernel();
	double* construct_test_kernel();
	void* train(double* K);
	double predict(double* test_K, int* test_labels);
	void write_files();
	void write_test_kernel();
	double* load_kernel(std::string kernel_name);

};

#endif