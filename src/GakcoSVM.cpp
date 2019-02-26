// GaKCo : Fast Gapped k-mer string Kernel using Counting
// Code Contibution by:
//Ritambhara Singh <rs3zz@virginia.edu>
//Kamran Kowsari <kk7nc@virginia.edu >
//Arshdeep Sekhon <as5cu@virginia.edu >
//Eamon Collins <ec3bd@virginia.edu>

/**
A tool wrapping kernel calculation and SVM calculation into one seamless object
**/
#include <stdio.h>
#include <cstdlib>
#include <string.h>
#include <math.h>
#include "shared.h"
#include <assert.h>
#include <thread>
#include <map>
#include <iostream>
#include <random>
#include <ctime>
#include "readInput.h"
#include <fstream>
#include <future>
#include "GakcoSVM.h"
#include "Gakco.h"
#define ARG_REQUIRED 7
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))


double* gakco_kernel_matrix  = NULL;
/*struct svm_parameter param;      // set by parse_command_line
struct svm_problem prob;         // set by read_problem
struct svm_model *model;
struct svm_node *x_space;*/
int cross_validation;
int nr_fold;

GakcoSVM::GakcoSVM(gakco_param* arg){
	this->params = arg;
}

double* GakcoSVM::construct_kernel(){

	char filename[100],filename_label[100],Dicfilename[100], opfilename[100];
	int *label;
	int k, max_m;
	int g;
	int na;
	long int nStr;
	
	int nfeat;
	double *K;
	unsigned int *nchoosekmat; //not all freed atm
	unsigned int *Ksfinal;
	int *len;
	int **S;
	int *feat;
	int *elems;
	long int maxlen, minlen;
	char isVerbose;
	Features *features;
	int numThreads;
	isVerbose = 0;

	strcpy(filename, this->params->filename.c_str());
	strcpy(Dicfilename, this->params->dictFilename.c_str());
	strcpy(filename_label, this->params->labelFilename.c_str());
	strcpy(opfilename, this->params->outputFilename.c_str());
	g = this->params->g;
	k = this->params->k;
	numThreads = this->params->threads;

	label = (int *) malloc(MAXNSTR * sizeof(int));
	len = (int *) malloc(MAXNSTR * sizeof(int));
	assert(len != 0);  
	maxlen = 0;
	minlen = STRMAXLEN;
	nStr = MAXNSTR;
	
	// Read input file
	if(!this->params->quiet)
		printf("Input file : %s\n", filename);
	S = Readinput_(filename, Dicfilename, label, len, &nStr, &maxlen, &minlen, &na, this);
	
	
	if (k <= 0 || g <= k || g>20 || g - k>20 || na <= 0){
		help();
		exit(1);
	}
	if(!this->params->quiet){
		if (maxlen != minlen)
			printf("Read %ld strings of max length = %ld and min length=%ld\n", nStr, maxlen, minlen);
		else
			printf("Read %ld strings of length = %ld\n", nStr, maxlen);
	}

	if (g > minlen){
		errorID1();
		exit(1);
	}



	/*Extract g-mers.*/
	features = extractFeatures(S, len, nStr, g);

	//now we can free the strings because we have the features
	for(int i = 0; i < nStr; i++){
		if(S[i] != NULL)
			free(S[i]);
	}
	free(len);
	free(S);
	
	nfeat = (*features).n;
	feat = (*features).features;
	if(!this->params->quiet)
		printf("(%d,%d): %d features\n", g, k, nfeat); 

	//number of 
	int num_str_pairs = nStr * (nStr+1) / 2;


	/* Precompute weights hm.*/

	int w[g - k];
	if(!this->params->quiet)
		printf("Weights (hm):");
	for (int i = 0; i <= g - k; i++) {
			w[i] = nchoosek(g - i, k);
			if(!this->params->quiet)
				printf("%d ", w[i]);
		}
	if(!this->params->quiet)
		printf("\n");
	

	//if this gakco is loading a kernel we don't need to calculate it
	if(this->params->loadkernel || this->params->loadmodel){
		this->kernel_features = features;
		this->nStr = nStr;
		this->labels = label;
		this->load_kernel(this->params->outputFilename);
		gakco_kernel_matrix = this->kernel;//for the svm train to access it
		return this->kernel;
	}

	/*Compute gapped kernel.*/
	K = (double *)malloc(num_str_pairs * sizeof(double));
	memset(K, 0, num_str_pairs*sizeof(double));


	
	//Ksfinal = (unsigned int **)malloc((g-k+1) * sizeof(unsigned int*));
		
	// Ksfinal = (unsigned int*)malloc(num_str_pairs * sizeof(unsigned int));
	// memset(Ksfinal, 0, sizeof(unsigned int) * num_str_pairs);

	
	
	elems = (int *)malloc(g * sizeof(int));
	
	for (int i = 0; i < g; ++i) {
		elems[i] = i;
	}

	max_m = g - k;

	//Create the work queue used for distributing tasks among threads
	int queueSize = 0;
	for (int m = max_m; m <= max_m; m++) {
		queueSize += nchoosek(g, m);
	}
	WorkItem *workQueue = new WorkItem[queueSize];
	int itemNum = 0;
	for (int m = max_m; m <= max_m; m++) {
		int numCombinations = nchoosek(g, m);
		for (int combNum = 0; combNum < numCombinations; combNum++) {
			workQueue[itemNum].m = m;
			workQueue[itemNum].combo_num = combNum;
			itemNum++;
		}
	}
	//Determine how many threads will be used
	if (numThreads == -1) {
		int numCores = std::thread::hardware_concurrency();
		numThreads = (numCores > 20) ? 20 : numCores;
	} else {
		numThreads = (numThreads > queueSize) ? queueSize : numThreads;
	}
	//Create an array of mutex locks (one for each value of m)
	pthread_mutex_t *mutexes = (pthread_mutex_t *) malloc((max_m + 1) * sizeof(pthread_mutex_t));
	for (int i = 0; i <= max_m; i++) {
		pthread_mutex_init(&mutexes[i], NULL);
	}
	//Create the threads and compute cumulative mismatch profiles
	if(!this->params->quiet)
		printf("Computing mismatch profiles using %d threads...\n", numThreads);
	std::vector<std::thread> threads;
	for (int i = 0; i < numThreads; i++) {
		threads.push_back(std::thread(&build_cumulative_mismatch_profiles_tri, workQueue, queueSize, i, numThreads,
			elems, features, K, feat, g, na, nfeat, nStr, mutexes, this->params->quiet));
	}
	for(auto &t : threads) {
		t.join();
	}
	if(!this->params->quiet)
		printf("\n");

	//no longer need the features list so can free it here
	free(features->features);
	free(features->group);
	free(features);
	
	// hm coefficients
	nchoosekmat = (unsigned int *) malloc(g * g * sizeof(unsigned int));
	memset(nchoosekmat, 0, sizeof(unsigned int) * g * g);
	
	// for ( int i = g; i >= 0; --i) {
	// 	for ( int j = 1; j <= i; ++j) {
	// 		nchoosekmat[(i - 1) + (j - 1)*g] = nchoosek(i, j);
	// 	}
	// }


	// for (int i = 1; i <= max_m; ++i) {
	// 	for (int j = 0; j <= i - 1; ++j) {
	// 		for (int j1 = 0; j1 < nStr; ++j1) {
	// 			for (int j2 = 0; j2 <= j1; ++j2) {
	// 				tri_access(Ksfinal[i], j1, j2) -= nchoosekmat[(g - j - 1) + (i - j - 1)*g] * tri_access(Ksfinal[j], j1, j2);
	// 			}
	// 		}
	// 	}
	// }

	// for (int j1 = 0; j1 < nStr; ++j1) {
	// 	for (int j2 = 0; j2 <= j1; ++j2) {
	// 		tri_access(K, j1, j2) += tri_access(Ksfinal, j1, j2);
	// 	}
	// }
	

	

	for(int i = 0; i < nStr; i++){
		for (int j = 0; j < i; j++){
			tri_access(K, i, j) = tri_access(K,i,j) / sqrt(tri_access(K,i,i) * tri_access(K,j,j));
		}
	}
	for(int i = 0; i < nStr; i++){
		tri_access(K,i,i) = 1.0;
	}


	// for(int i=0; i <= max_m; i++){
	// 	free(Ksfinal[i]);
	// }
	free(Ksfinal);
	free(nchoosekmat);
	//free(feat); //can't free it cause we need it later for test kernel generation
	free(elems);
	//free(S); //not sure why this wasn't here, include later if not usefull

	this->kernel_features = features;
	this->kernel = K;
	gakco_kernel_matrix = K;//for the svm train to access it
	this->nStr = nStr;
	this->labels = label;
	return K;
}

double* GakcoSVM::construct_test_kernel(){
	if (this->kernel == NULL) {
		printf("Must calculate train kernel before test kernel");
		return NULL;
	} else if (this->model == NULL){
		printf("Must train SVM before constructing the test kernel");
		return NULL;
	}

	int ** S;
	long int nTestStr;
	int *label, *len;
	int k, max_m;
	int g, numThreads;
	int na;
	long int maxlen, minlen;
	int *elems;
	Features *features;
	double *test_K;
	unsigned int* nchoosekmat;
	unsigned int* test_Ksfinal;
	
	g = this->params->g;
	k = this->params->k;
	numThreads = this->params->threads;

	int* test_label = (int *)malloc(MAXNSTR * sizeof(int));
	int* test_len = (int *)malloc(MAXNSTR * sizeof(int));
	long int test_maxlen = 0;
	long int test_minlen = STRMAXLEN;
	nTestStr = MAXNSTR;
	int test_na;

	label = (int *)malloc(MAXNSTR * sizeof(int));
	len = (int *)malloc(MAXNSTR * sizeof(int));
	maxlen = 0;
	minlen = STRMAXLEN;
	nStr = MAXNSTR;

	//calculate the total number of support vectors in the model
	int num_sv = this->model->nSV[0] + this->model->nSV[1];

	//reading input from test file
	int** test_S = Readinput_(&(this->params->testFilename)[0],&(this->params->dictFilename)[0],test_label,test_len, &nTestStr, &test_maxlen, &test_minlen,&test_na, this);
	this->nTestStr = nTestStr;
	this->test_labels = test_label;

	if (k <= 0 || g <= k || g>20 || g - k>20 || test_na <= 0){
		help();
		exit(1);
	}
	if(!this->params->quiet){
		if (maxlen != minlen)
			printf("Read %ld strings of max length = %ld and min length=%ld\n", nTestStr, test_maxlen, test_minlen);
		else
			printf("Read %ld strings of length = %ld\n", nTestStr, test_maxlen);
	}

	if (g > minlen){
		errorID1();
		exit(1);
	}


	//read train data
	S = Readinput_(&(this->params->filename)[0],&(this->params->dictFilename)[0],label,len, &nStr, &maxlen, &minlen,&na, this);

	
	int totalStr = nTestStr + num_sv;
	int** finalS = (int**)malloc(totalStr * sizeof(int*));
	//create a unified length array
	int* finalLen = (int*)malloc(totalStr * sizeof(int));
	memset(finalLen, 0, totalStr*sizeof(int));
	
	//copy in the references to the test strings
	memcpy(finalS, test_S, nTestStr * sizeof(int*));
	memcpy(finalLen, test_len, nTestStr*sizeof(int));

	//copy the references to support vector strings, free those that aren't SVs
	int svcount = 0;
	for(int i = 0; i < nStr; i++){
		int svind = this->model->sv_indices[svcount] -1;
		if(i == svind){
			finalS[nTestStr + svcount] = S[i];
			finalLen[nTestStr + svcount] = len[i];
			svcount++;
		}else{
			free(S[i]);
			S[i] = NULL;
		}
	}


	features = extractFeatures(finalS, finalLen, totalStr, g);

	//now we can free the strings because we have the features
	for(int i = 0; i < nStr; i++){
		if(S[i] != NULL)
			free(S[i]);
	}
	for(int i = 0; i < nTestStr; i++){
		free(test_S[i]);
	}
	free(test_len);
	free(len);
	free(finalLen);
	free(S);
	free(test_S);
	free(finalS);


	/* Precompute weights hm.*/

	int w[g - k];
	if(!this->params->quiet)
		printf("Weights (hm):");
	for (int i = 0; i <= g - k; i++){
		w[i] = nchoosek(g - i, k);
		if(!this->params->quiet)
			printf("%d ", w[i]);
	}
	

	
	int tri_totalStr = totalStr * (totalStr+1) / 2;

	//malloc things we have the size info on already here so there isn't excessive mallocing inside the loop
	//test kernel is a non-triangular matrix of dim nTestStr x nSV
	test_K = (double *)malloc(tri_totalStr * sizeof(double));
	//malloc test_Ksfinal here, memset it each time we use it tho
	//test_Ksfinal = (unsigned int **)malloc((g - k + 1) * sizeof(unsigned int*));
	
	test_Ksfinal = (unsigned int*)malloc(tri_totalStr * sizeof(unsigned int));
	memset(test_Ksfinal, 0, tri_totalStr * sizeof(unsigned int));
	
	elems = (int *)malloc(g * sizeof(int));
	for (int i = 0; i < g; ++i){
		elems[i] = i;
	}
	nchoosekmat = (unsigned int *)malloc(g*g * sizeof(unsigned int));

	

	//memset test_K before we use it
	memset(test_K, 0, sizeof(double) * tri_totalStr);
	memset(nchoosekmat, 0, sizeof(unsigned int) * g * g);


	max_m = g - k;

	//Create the work queue used for distributing tasks among threads
	int queueSize = 0;
	for (int m = 0; m <= max_m; m++) {
		queueSize += nchoosek(g, m);
	}
	WorkItem *workQueue = new WorkItem[queueSize];
	int itemNum = 0;
	for (int m = 0; m <= max_m; m++) {
		int numCombinations = nchoosek(g, m);
		for (int combNum = 0; combNum < numCombinations; combNum++) {
			workQueue[itemNum].m = m;
			workQueue[itemNum].combo_num = combNum;
			itemNum++;
		}
	}
	//Determine how many threads will be used
	if (numThreads == -1) {
		int numCores = std::thread::hardware_concurrency();
		numThreads = (numCores > 20) ? 20 : numCores;
	} else {
		numThreads = (numThreads > queueSize) ? queueSize : numThreads;
	}
	//Create an array of mutex locks (one for each value of m)
	pthread_mutex_t *mutexes = (pthread_mutex_t *) malloc((max_m + 1) * sizeof(pthread_mutex_t));
	for (int i = 0; i <= max_m; i++) {
		pthread_mutex_init(&mutexes[i], NULL);
	}
	//Create the threads and compute cumulative mismatch profiles
	if(!this->params->quiet)
		printf("Computing mismatch profiles using %d threads...\n", numThreads);
	std::vector<std::thread> threads;
	for (int i = 0; i < numThreads; i++) {
		threads.push_back(std::thread(&build_cumulative_mismatch_profiles_tri, workQueue, queueSize, i, numThreads,
			elems, features, test_K, features->features, g, na, features->n, totalStr, mutexes, this->params->quiet));
	}
	for(auto &t : threads) {
		t.join();
	}
	if(!this->params->quiet)
		printf("\n");

	//no longer need the features list so can free it here
	free(features->features);
	free(features->group);
	free(features);
	
	// hm coefficients
	nchoosekmat = (unsigned int *) malloc(g * g * sizeof(unsigned int));
	memset(nchoosekmat, 0, sizeof(unsigned int) * g * g);
	
	for ( int i = g; i >= 0; --i) {
		for ( int j = 1; j <= i; ++j) {
			nchoosekmat[(i - 1) + (j - 1)*g] = nchoosek(i, j);
		}
	}


	// for (int j1 = 0; j1 < totalStr; ++j1) {
	// 	for (int j2 = 0; j2 <= j1; ++j2) {
	// 		//test_K[j1 + j2*totalStr] += w[i] * test_Ksfinal[(c1 + j1) + j2*totalStr];
	// 		tri_access(test_K, j1, j2) += tri_access(test_Ksfinal, j1, j2);
	// 	}
	// }

	//free test_Ksfinal before allocating K so total memory usage at any given point is reduced
	// for(int i = 0; i <= max_m; i++){
	// 	free(test_Ksfinal[i]);
	// }
	//free(test_Ksfinal);


	double* K = (double*)malloc(nTestStr * num_sv * sizeof(double));
	
	for(int i = 0; i < nTestStr; i++){
		for(int j = nTestStr; j < totalStr; j++){
			K[i*num_sv + j - nTestStr] = tri_access(test_K, i, j) / sqrt(tri_access(test_K, i, i) * tri_access(test_K, j, j));
		}
	}


	free(test_K);


	this->test_kernel = K;
	return K;
}

void* GakcoSVM::construct_linear_kernel(){
	int ** S;
	long int nTestStr;
	int *label, *len;
	int k, max_m;
	int g, numThreads;
	int na;
	long int maxlen, minlen;
	int *elems;
	Features *features;
	double *total_K;
	unsigned int* nchoosekmat;
	unsigned int* total_Ksfinal;
	
	g = this->params->g;
	k = this->params->k;
	numThreads = this->params->threads;

	int* test_label = (int *)malloc(MAXNSTR * sizeof(int));
	int* test_len = (int *)malloc(MAXNSTR * sizeof(int));
	long int test_maxlen = 0;
	long int test_minlen = STRMAXLEN;
	nTestStr = MAXNSTR;
	int test_na;

	label = (int *)malloc(MAXNSTR * sizeof(int));
	len = (int *)malloc(MAXNSTR * sizeof(int));
	maxlen = 0;
	minlen = STRMAXLEN;
	nStr = MAXNSTR;


	//read train data
	S = Readinput_(&(this->params->filename)[0],&(this->params->dictFilename)[0],label,len, &nStr, &maxlen, &minlen,&na, this);
	this->nStr = nStr;
	this->labels = label;

	//reading input from test file
	int** test_S = Readinput_(&(this->params->testFilename)[0],&(this->params->dictFilename)[0],test_label,test_len, &nTestStr, &test_maxlen, &test_minlen,&test_na, this);
	this->nTestStr = nTestStr;
	this->test_labels = test_label;

	if (k <= 0 || g <= k || g>20 || g - k>20 || na <= 0){
		help();
		exit(1);
	}
	if(!this->params->quiet){
		if (maxlen != minlen)
			printf("Read %ld strings of max length = %ld and min length=%ld\n", nStr+nTestStr, std::max(test_maxlen, maxlen), std::min(test_minlen, minlen));
		else
			printf("Read %ld strings of length = %ld\n", nStr+nTestStr, std::max(test_maxlen,maxlen));
	}
	if (g > minlen){
		errorID1();
		exit(1);
	}


	
	int totalStr = nTestStr + nStr;
	int** finalS = (int**)malloc(totalStr * sizeof(int*));
	//create a unified length array
	int* finalLen = (int*)malloc(totalStr * sizeof(int));
	
	//copy the references to train strings
	memcpy(finalS, S, nStr * sizeof(int*));
	memcpy(finalLen, len, nStr * sizeof(int));

	//copy in the references to the test strings
	memcpy(&finalS[nStr], test_S, nTestStr * sizeof(int*));
	memcpy(&finalLen[nStr], test_len, nTestStr*sizeof(int));



	features = extractFeatures(finalS, finalLen, totalStr, g);

	//now we can free the strings because we have the features
	for(int i = 0; i < nStr; i++){
		free(S[i]);
	}
	for(int i = 0; i < nTestStr; i++){
		free(test_S[i]);
	}
	free(test_len);
	free(len);
	free(finalLen);
	free(S);
	free(test_S);
	free(finalS);


	/* Precompute weights hm.*/

	int w[g - k];
	if(!this->params->quiet)
		printf("Weights (hm):");
	for (int i = 0; i <= g - k; i++){
		w[i] = nchoosek(g - i, k);
		if(!this->params->quiet)
			printf("%d ", w[i]);
	}
	
		
	int tri_totalStr = totalStr * (totalStr+1) / 2;

	
	elems = (int *)malloc(g * sizeof(int));
	for (int i = 0; i < g; ++i){
		elems[i] = i;
	}
	nchoosekmat = (unsigned int *)malloc(g*g * sizeof(unsigned int));

	
	//malloc things we have the size info on already here so there isn't excessive mallocing inside the loop
	//test kernel is a non-triangular matrix of dim nTestStr x nSV
	total_K = (double *)malloc(tri_totalStr * sizeof(double));
	//memset K before we use it
	memset(total_K, 0, sizeof(double) * tri_totalStr);
	memset(nchoosekmat, 0, sizeof(unsigned int) * g * g);


	max_m = g - k;

	//Create the work queue used for distributing tasks among threads
	int queueSize = nchoosek(g, max_m);

	WorkItem *workQueue = new WorkItem[queueSize];
	int itemNum = 0;
	for (int combNum = 0; combNum < queueSize; combNum++) {
		workQueue[itemNum].m = max_m;
		workQueue[itemNum].combo_num = combNum;
		itemNum++;
	}
	
	//Determine how many threads will be used
	if (numThreads == -1) {
		int numCores = std::thread::hardware_concurrency();
		numThreads = (numCores > 20) ? 20 : numCores;
	}
	numThreads = (numThreads > queueSize) ? queueSize : numThreads;
	
	//Create an array of mutex locks (one for each value of m)
	pthread_mutex_t *mutex = (pthread_mutex_t *) malloc(sizeof(pthread_mutex_t));
	pthread_mutex_init(&mutex[0], NULL);

	//Create the threads and compute cumulative mismatch profiles
	if(!this->params->quiet)
		printf("Computing mismatch profiles using %d threads...\n", numThreads);
	std::vector<std::thread> threads;
	for (int i = 0; i < numThreads; i++) {
		threads.push_back(std::thread(&build_cumulative_mismatch_profiles_tri, workQueue, queueSize, i, numThreads,
			elems, features, total_K, features->features, g, na, features->n, totalStr, mutex, this->params->quiet));
	}
	for(auto &t : threads) {
		t.join();
	}
	if(!this->params->quiet)
		printf("\n");

	//no longer need the features list so can free it here
	free(features->features);
	free(features->group);
	free(features);
	
	// hm coefficients
	nchoosekmat = (unsigned int *) malloc(g * g * sizeof(unsigned int));
	memset(nchoosekmat, 0, sizeof(unsigned int) * g * g);
	

//TODO switch to memcpy which it's now eligible for as we aren't actually aggregating anymore.
	// for (int j1 = 0; j1 < totalStr; ++j1) {
	// 	for (int j2 = 0; j2 <= j1; ++j2) {
	// 		//test_K[j1 + j2*totalStr] += w[i] * test_Ksfinal[(c1 + j1) + j2*totalStr];
	// 		tri_access(total_K, j1, j2) += tri_access(total_Ksfinal, j1, j2);
	// 	}
	// }


	//free test_Ksfinal before allocating K so total memory usage at any given point is reduced
	// for(int i = 0; i <= max_m; i++){
	// 	free(total_Ksfinal[i]);
	// }
	//free(total_Ksfinal);


	double* test_K = (double*)malloc(nTestStr * nStr * sizeof(double));
	
	for(int i = nStr; i < totalStr; i++){
		for(int j = 0; j < nStr; j++){
			test_K[(i-nStr)*nStr + j] = tri_access(total_K, i, j) / sqrt(tri_access(total_K, i, i) * tri_access(total_K, j, j));
		}
	}

	//if loadkernel is 1, we already have a train kernel and don't need this extra computation
	if(!this->params->loadkernel){
		//reallocate to a smaller size to encapsulate only the train set v train set
		double* K = (double*)realloc(total_K, nStr*(nStr+1) / 2 * sizeof(double));

		for(int i = 0; i < nStr; i++){
			printf("%f \n",tri_access(total_K,i,i));
		}

		// for(int i = 0; i < nStr; i++){
		// 	for(int j = 0; j < i; j++){
		// 		tri_access(K, i, j) = tri_access(total_K, i, j) / sqrt(tri_access(total_K, i, i) * tri_access(total_K, j, j));
		// 	}
		// }



		this->kernel = K;
	}else{
		free(total_K);
	}



	this->test_kernel = test_K;

	return this->kernel;
}


void GakcoSVM::train(double* K) {
	if(K == NULL){
		K = this->kernel;
	}
	struct svm_parameter* svm_param = Malloc(svm_parameter, 1);
	struct svm_problem* prob = Malloc(svm_problem, 1);

	this->prob = prob;
	

	const char* error_msg;

	//pass the values from the main params struct in. Will be defaults or whatever
	//the user has initialized them to. Params can be changed in between different training runs on the same object
	svm_param->svm_type = this->params->svm_type;
	svm_param->kernel_type = this->params->kernel_type;
	svm_param->nu = this->params->nu;
	svm_param->cache_size = this->params->cache_size;
	svm_param->C = this->params->C;
	svm_param->nr_weight = this->params->nr_weight;
	svm_param->weight_label = this->params->weight_label;
	svm_param->weight = this->params->weight;
	svm_param->shrinking = this->params->h;
	svm_param->probability = this->params->probability;
	svm_param->eps = this->params->eps;
	svm_param->degree = 0;


	struct svm_node** x;
	svm_node* x_space;

	prob->l = this->nStr;
	prob->y = Malloc(double, prob->l);
	x = Malloc(svm_node*, prob->l);

	if(this->params->kernel_type == GAKCO){
		x_space = Malloc(struct svm_node, prob->l); // Kind-of hacky, but we're just going to have 1 node per thing.

		for (int i = 0; i < prob->l; i++){
			x_space[i].index = i;
			x_space[i].value = i;
			x[i] = &x_space[i];
			prob->y[i] = this->labels[i];

		}
		this->x_space = x_space;
	}else if(this->params->kernel_type == LINEAR){

		x_space = Malloc(struct svm_node, (nStr+1)*nStr);
		int totalind = 0;
		for (int i = 0; i < nStr; i++){
			x[i] = &x_space[totalind];
			for(int j = 0; j < nStr; j++){
				x_space[j+i*(nStr+1)].index = j+1; 
				x_space[j+i*(nStr+1)].value = tri_access(K, i, j);
			}
			totalind += nStr;
			x_space[totalind].index = -1;
			totalind++;
			prob->y[i] = this->labels[i];
		}
		this->x_space = x_space;
	}

	prob->x = x;

	//if in quiet mode, set libsvm's print function to null
	if(this->params->quiet)
		svm_set_print_string_function(&print_null);

	error_msg = svm_check_parameter(prob, svm_param);

	if (error_msg) {
		fprintf(stderr, "ERROR: %s\n", error_msg);
		exit(1);
	}

	//train that ish
	this->model = svm_train(prob, svm_param);

	if (!this->params->modelName.empty()){
		svm_save_model(this->params->modelName.c_str(), this->model);
	}

	free(svm_param);
	//free(x_space);
	//free(x);
	//free(prob);
	free(this->kernel);

}



//file output name specified via command line or specially by modifying the parameter struct
void GakcoSVM::write_files() {
	FILE *kernelfile;
	FILE *labelfile;
	std::string kernelfileName = this->params->outputFilename;
	if(kernelfileName.empty()){
		kernelfileName = "kernel.txt";
	}
	if(!this->params->quiet)
		printf("Writing kernel to %s\n", kernelfileName.c_str());
	kernelfile = fopen(kernelfileName.c_str(), "w");
	labelfile = fopen("train_labels.txt", "w");
	int nStr = this->nStr;

	for (int i = 0; i < nStr; ++i) {	
		for (int j = 0; j <= i; ++j) {
			fprintf(kernelfile, "%d:%e ", j + 1, tri_access(this->kernel,i,j));
		}
		fprintf(kernelfile, "\n");
		fprintf(labelfile, "%d\n", this->labels[i]);
	}
	fclose(kernelfile);
	fclose(labelfile);
}

void GakcoSVM::write_dictionary(char* dictionary){
	FILE *dictfile;
	dictfile = fopen(this->params->dictFilename.c_str(), "w");
	for(int i= 0; i < strlen(dictionary); i++){
		fprintf(dictfile, "%c\n", dictionary[i]);
	}
	fclose(dictfile);
}

//outputs a kernel compatible for use with svm-train executable
void GakcoSVM::write_libsvm_kernel() {
	FILE *kernelfile;
	FILE *labelfile;
	std::string kernelfileName = this->params->outputFilename;
	if(kernelfileName.empty()){
		kernelfileName = "kernel.txt";
	}
	if(!this->params->quiet)
		printf("Writing kernel to %s\n", kernelfileName.c_str());
	kernelfile = fopen(kernelfileName.c_str(), "w");
	labelfile = fopen("train_labels.txt", "w");
	int nStr = this->nStr;

	for (int i = 0; i < nStr; ++i) {
		fprintf(kernelfile, "%d ", this->labels[i]);	
		for (int j = 0; j < nStr; ++j) {
			fprintf(kernelfile, "%d:%e ", j + 1, tri_access(this->kernel,i,j));
		}
		fprintf(kernelfile, "\n");
		fprintf(labelfile, "%d\n", this->labels[i]);
	}
	fclose(kernelfile);
	fclose(labelfile);
}

void GakcoSVM::write_test_kernel() {
	FILE *kernelfile;
	FILE *labelfile;
	kernelfile = fopen("test_Kernel.txt", "w");
	labelfile = fopen(this->params->labelFilename.c_str(), "w");
	int nStr = this->nStr;
	int nTestStr = this->nTestStr;
	int num_sv = this->model->nSV[0] + this->model->nSV[1];

	if(!this->params->quiet)
		printf("Writing test kernel %d", nTestStr);
	for (int i = 0; i < nTestStr; ++i)
	{	
		//fprintf(kernelfile, "%d ", this->test_labels[i]);
		if(this->params->kernel_type == GAKCO){
			for (int j = 0; j < num_sv; j++)
			{
				fprintf(kernelfile, "%d:%.12f ", this->model->sv_indices[j], this->test_kernel[j + i*num_sv]);
			}
		}else if (this->params->kernel_type == LINEAR){
			for (int j = 0; j < nStr; j++)
			{
				fprintf(kernelfile, "%d:%.12f ", j+1, this->test_kernel[j + i*nStr]);
			}
		}
		fprintf(labelfile, "%d\n", this->test_labels[i]);
		fprintf(kernelfile, "\n");
	}
		
	fclose(kernelfile);
	fclose(labelfile);
}



double GakcoSVM::predict(double *test_K, int* test_labels){
	int nStr = this->nStr;
	int num_sv = this->model->nSV[0] + this->model->nSV[1];
	int nTestStr = this->nTestStr;


	struct svm_node *x = Malloc(struct svm_node, nStr + 1);
	int correct = 0;
	int pagg =0, nagg=0; //aggregators for finding num of pos and neg samples for auc
	double* neg = Malloc(double, nTestStr);
	double* pos = Malloc(double, nTestStr);
	int fp = 0, fn = 0; //counters for false postives and negatives
	int labelind = 0;
	for (int i =0; i < 2; i++){
		if (this->model->label[i] == 1)
			labelind = i;
	}

	FILE* labelfile;
	labelfile = fopen(this->params->labelFilename.c_str(), "w");

	int svcount = 0;
	for(int i = 0; i < nTestStr; i++){
		if(this->params->kernel_type == GAKCO){
			for (int j=0; j < num_sv; j++){
				x[j].index = this->model->sv_indices[j];
				x[j].value = test_K[i * num_sv + j];
				svcount++;
			}
			x[num_sv].index = -1;
		}else if(this->params->kernel_type == LINEAR){
			for(int j=0; j < nStr; j++){
				x[j].index = j+1;
				x[j].value = test_K[i*nStr +j];
			}
			x[nStr].index = -1;
		}

		double probs[2];
		double guess = svm_predict_probability(this->model, x, probs);
		//double guess = svm_predict_values(this->model, x, probs);
		

		if (test_labels[i] > 0){
			pos[pagg] = probs[labelind];
			pagg += 1;
			if(guess < 0)
				fn++;
		} 
		else{
			neg[nagg] = probs[labelind];
			nagg += 1;
			if(guess > 0)
				fp++;
		} 


		fprintf(labelfile, "%d\n", (int)guess);
	
		if ((guess < 0.0 && test_labels[i] < 0) || (guess > 0.0 && test_labels[i] > 0)){
			correct++;
		}
	}

	
	printf("\nacc: %f\n", (double)correct / nTestStr);

	if(this->params->probability && this->numClasses){
		double auc = calculate_auc(pos, neg, pagg, nagg);
		printf("auc: %f\n", auc);
		if(!this->params->quiet){
			printf("fp: %d\tfn: %d\n", fp, fn);
			printf("num pos: %d\n", pagg);
			printf("percent pos: %f\n", ((double)pagg/(nagg+pagg)));
		}
	}

	fclose(labelfile);
	free(pos);
	free(neg);
	free(test_K);
	free(x);
	// if(this->params->kernel_type == GAKCO){
	// 	for (int i = 0; i < this->prob->l; i++){
	// 		free(this->prob->x[i]);
	// 	}
	// }
	// else{
	// 	free(this->x_space);
	// }
	free(this->x_space);
	free(this->prob->x);
	free(this->prob->y);
	free(this->prob);

	free(this->labels);
	free(this->test_labels);

	//model freeing
	free(this->model);
	free(this->model->SV);
	for(int i=0;i<this->model->nr_class-1;i++)
		free(model->sv_coef[i]);
	free(model->sv_coef);
	free(model->sv_indices);


	return (double)correct / nTestStr;
}

double calculate_auc(double* pos, double* neg, int npos, int nneg){
	int correct = 0;
	int total = 0;
	for (int i = 0; i < npos; i++){
		for (int j = 0; j < nneg; j++){
			if (pos[i] > neg[j]){
				correct++;
			}	
			total++;
		}
	}
	return (double)correct / total;
}

double *GakcoSVM::load_kernel(std::string kernel_name){

	std::string line;
	std::ifstream inpfile (kernel_name);
	double* K = NULL;
	int lines = 0;
	if (inpfile.is_open()){
		while(!inpfile.eof()){
			std::getline(inpfile, line);
			lines++;
		}
		//int width = find last number for test kernel loading too
		K = (double*) malloc(lines * (lines+1) / 2 * sizeof(double));
		//return the cursor to the beginning of the file
		inpfile.clear();
		inpfile.seekg(0, std::ios::beg);
		int idx = 0;
		while(!inpfile.eof()){
			std::getline(inpfile, line, ' ');
			line = trim(line);
			//printf("%s\n",line.c_str());
			if(line.empty() || line==std::string("\n"))
				continue;
			K[idx] = std::stof(line.substr(line.find(":") + 1)); //get trimmed value without index and colon, convert to double
			idx++;
		}
		inpfile.close();
	}


	if(this->kernel != NULL)
		free(this->kernel);
	this->kernel = K;

	return K;
}	