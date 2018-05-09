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
	int k, num_max_mismatches, max_m;
	int m, g;
	int na;
	unsigned int addr;
	long int nStr, num_comb, value;
	
	int nfeat;
	double *K;
	unsigned int *nchoosekmat, *Ks, *Ksfinal, *Ksfinalmat; //not all freed atm
	int *len;
	int **S;
	unsigned int *sortIdx;
	int *feat;
	unsigned int  *out, *out_temp, *resgroup;
	int *elems, *cnt_k;
	unsigned int *cnt_m;
	long int maxIdx, maxlen, minlen;
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
	
	printf("Input file : %s\n", filename);
	S = Readinput_(filename, Dicfilename, label, len, &nStr, &maxlen, &minlen, &na);
	
	
	if (k <= 0 || g <= k || g>20 || g - k>20 || na <= 0){
		help();
		exit(1);
	}
	if (maxlen != minlen)
		printf("Read %ld strings of max length = %ld and min length=%ld\n", nStr, maxlen, minlen);
	else
		printf("Read %ld strings of length = %ld\n", nStr, maxlen);

	if (g > minlen){
		errorID1();
		exit(1);
	}



	/*Extract g-mers.*/
	features = extractFeatures(S, len, nStr, g);
	
	nfeat = (*features).n;
	feat = (*features).features;
	printf("(%d,%d): %d features\n", g, k, nfeat); 

	//number of 
	int num_str_pairs = nStr * (nStr+1) / 2;


	/* Precompute weights hm.*/

	int w[g - k];
	printf("Weights (hm):");
	for (int i = 0; i <= g - k; i++) {
			w[i] = nchoosek(g - i, k);
			printf("%d ", w[i]);
		}
	
	printf("\n");

	//if this gakco is loading a kernel we don't need to calculate it
	if(this->params->loadkernel || this->params->loadmodel){
		this->kernel_features = features;
		//gakco_kernel_matrix = K;//for the svm train to access it
		this->nStr = nStr;
		this->labels = label;
		this->load_kernel(this->params->outputFilename);
		return this->kernel;
	}

	/*Compute gapped kernel.*/
	K = (double *)malloc(nStr*nStr * sizeof(double));

	

	addr = ((g - k) + 1)*nStr*nStr;
	
	Ksfinal = (unsigned int *)malloc(addr * sizeof(unsigned int));
	
	memset(Ksfinal, 0, sizeof(unsigned int) * addr);
	
	elems = (int *)malloc(g * sizeof(int));
	
	cnt_k = (int *)malloc(nfeat * sizeof(int));
	for (int i = 0; i < g; ++i) {
		elems[i] = i;
	}

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
	printf("Computing mismatch profiles using %d threads...\n", numThreads);
	std::vector<std::thread> threads;
	for (int i = 0; i < numThreads; i++) {
		threads.push_back(std::thread(&build_cumulative_mismatch_profiles, workQueue, queueSize, i, numThreads,
			elems, features, Ksfinal, cnt_k, feat, g, na, nfeat, nStr, mutexes));
	}
	for(auto &t : threads) {
		t.join();
	}
	printf("\n");
	
	// hm coefficients
	nchoosekmat = (unsigned int *) malloc(g * g * sizeof(unsigned int));
	memset(nchoosekmat, 0, sizeof(unsigned int) * g * g);
	
	for ( int i = g; i >= 0; --i) {
		for ( int j = 1; j <= i; ++j) {
			nchoosekmat[(i - 1) + (j - 1)*g] = nchoosek(i, j);
		}
	}

	int c1 = 0,
	c2 = 0;

	for (int i = 1; i <= max_m; ++i) {
		c1 = cnt_k[i];
		for (int j = 0; j <= i - 1; ++j) {
			c2 = cnt_k[j];
			for (int j1 = 0; j1 < nStr; ++j1) {
				value = 0;
				int x = 0;
				for (int j2 = 0; j2 < nStr; ++j2) {
					Ksfinal[(c1 + j1) + j2*nStr] -=  nchoosekmat[(g - j - 1) + (i - j - 1)*g] * Ksfinal[(c2 + j1) + j2*nStr];
				}
			}
		}
	}
	for (int i = 0; i <= g - k; i++) {
		c1 = cnt_k[i];
		for (int j1 = 0; j1 < nStr; ++j1) {
			for (int j2 = 0; j2 < nStr; ++j2) {
				K[j1 + j2*nStr] += w[i] * Ksfinal[(c1 + j1) + j2*nStr];
			}
		}
	}

	for(int i = 0; i < nStr; i++){
		for (int j = 0; j < i; j++){
			double temp = K[i*nStr + j] / sqrt(K[i*nStr + i] * K[j*nStr + j]);
			K[i*nStr + j] = temp;
			K[j*nStr + i] = temp;
		}
	}
	for(int i = 0; i < nStr; i++){
		K[i*nStr + i] = 1.0;
	}

	free(cnt_k);
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
	int k, num_max_mismatches, max_m;
	int m, g, numThreads;
	int na;
	unsigned int addr;
	long int num_comb, value, maxlen, minlen;
	int *elems, *cnt_k;
	Features *features;
	double *test_K;
	unsigned int* test_Ksfinal, *nchoosekmat;
	
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
	int** test_S = Readinput_(&(this->params->testFilename)[0],&(this->params->dictFilename)[0],test_label,test_len, &nTestStr, &test_maxlen, &test_minlen,&test_na);
	this->nTestStr = nTestStr;
	this->test_labels = test_label;

	if (k <= 0 || g <= k || g>20 || g - k>20 || test_na <= 0){
		help();
		exit(1);
	}
	if (maxlen != minlen)
		printf("Read %ld strings of max length = %ld and min length=%ld\n", nTestStr, test_maxlen, test_minlen);
	else
		printf("Read %ld strings of length = %ld\n", nTestStr, test_maxlen);

	if (g > minlen){
		errorID1();
		exit(1);
	}


	//read train data
	S = Readinput_(&(this->params->filename)[0],&(this->params->dictFilename)[0],label,len, &nStr, &maxlen, &minlen,&na);

	
	int totalStr = nTestStr + num_sv;
	int** finalS = (int**)malloc(totalStr * sizeof(int*));
	//create a unified length array
	int* finalLen = (int*)malloc(totalStr * sizeof(int));
	
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
		}
	}

	//now we can free the length arrays and the arrays holding the references to the strings
	free(test_len);
	free(len);
	free(S);
	free(test_S);


	features = extractFeatures(finalS, finalLen, totalStr, g);


	/* Precompute weights hm.*/

	int w[g - k];
	printf("Weights (hm):");
	for (int i = 0; i <= g - k; i++){
		w[i] = nchoosek(g - i, k);
		printf("%d ", w[i]);
	}

	
	addr = ((g - k) + 1)*totalStr*totalStr;

	//malloc things we have the size info on already here so there isn't excessive mallocing inside the loop
	//test kernel is a non-triangular matrix of dim nTestStr x nSV
	test_K = (double *)malloc(totalStr * totalStr * sizeof(double));
	//malloc test_Ksfinal here, memset it each time we use it tho
	test_Ksfinal = (unsigned int *)malloc(addr * sizeof(unsigned int));
	elems = (int *)malloc(g * sizeof(int));
	for (int i = 0; i < g; ++i){
		elems[i] = i;
	}
	nchoosekmat = (unsigned int *)malloc(g*g * sizeof(unsigned int));

	

	//memset test_Ksfinal and test_K before we use it
	memset(test_Ksfinal, 0, sizeof(unsigned int) * addr);
	memset(test_K, 0, sizeof(double) * totalStr*totalStr);
	memset(nchoosekmat, 0, sizeof(unsigned int) * g * g);

	cnt_k = (int *)malloc(features->n * sizeof(int));

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
	printf("Computing mismatch profiles using %d threads...\n", numThreads);
	std::vector<std::thread> threads;
	for (int i = 0; i < numThreads; i++) {
		threads.push_back(std::thread(&build_cumulative_mismatch_profiles, workQueue, queueSize, i, numThreads,
			elems, features, test_Ksfinal, cnt_k, features->features, g, na, features->n, totalStr, mutexes));
	}
	for(auto &t : threads) {
		t.join();
	}
	printf("\n");
	
	// hm coefficients
	nchoosekmat = (unsigned int *) malloc(g * g * sizeof(unsigned int));
	memset(nchoosekmat, 0, sizeof(unsigned int) * g * g);
	
	for ( int i = g; i >= 0; --i) {
		for ( int j = 1; j <= i; ++j) {
			nchoosekmat[(i - 1) + (j - 1)*g] = nchoosek(i, j);
		}
	}

	int c1 = 0,
	c2 = 0;
	
	//get exact mismatch profile (remove the overcounting)
	
	for (int i = 1; i <= max_m; ++i) {
		c1 = cnt_k[i];
		for (int j = 0; j <= i - 1; ++j) {
			c2 = cnt_k[j];
			for (int j1 = 0; j1 < totalStr; ++j1) {
				value = 0;
				int x = 0;
				for (int j2 = 0; j2 < totalStr; ++j2) {
					test_Ksfinal[(c1 + j1) + j2*totalStr] -=  nchoosekmat[(g - j - 1) + (i - j - 1)*g] * test_Ksfinal[(c2 + j1) + j2*totalStr];
				}
			}
		}
	}
	for (int i = 0; i <= g - k; i++) {
		c1 = cnt_k[i];
		for (int j1 = 0; j1 < totalStr; ++j1) {
			for (int j2 = 0; j2 < totalStr; ++j2) {
				test_K[j1 + j2*totalStr] += w[i] * test_Ksfinal[(c1 + j1) + j2*totalStr];
			}
		}
	}


	double* K = (double*)malloc(nTestStr * num_sv * sizeof(double));
	
	for(int i = 0; i < nTestStr; i++){
		for(int j = nTestStr; j < totalStr; j++){
			K[i*num_sv + j - nTestStr] = test_K[i*totalStr + j] / sqrt(test_K[i*totalStr + i] * test_K[j*totalStr + j]);
		}
	}


	free(cnt_k);
	free(test_K);
	free(test_Ksfinal);


	this->test_kernel = K;
	return K;
}


void* GakcoSVM::train(double* K) {
	if(K == NULL){
		K = this->kernel;
	}
	struct svm_parameter* svm_param = Malloc(svm_parameter, 1);
	struct svm_problem* prob = Malloc(svm_problem, 1);
	struct svm_model *model;
	

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

	for (int i = 0; i < prob->l; i++){

		svm_node* x_space = Malloc(svm_node, prob->l + 1);

		for (int j = 0; j < prob->l; j++){
			x_space[j].index = j+1;
			x_space[j].value = K[i * prob->l + j];
		}

		x_space[prob->l].index = -1;
		x[i] = x_space;
		prob->y[i] = this->labels[i];

	}

	prob->x = x;



	error_msg = svm_check_parameter(prob, svm_param);

	if (error_msg) {
		fprintf(stderr, "ERROR: %s\n", error_msg);
		exit(1);
	}

	if (this->params->crossfold) {//not working yet
		return NULL;
        int total_correct = 0;
        double *target = Malloc(double, prob->l);

        svm_cross_validation(prob, svm_param, this->params->crossfold, target);
        for (int i = 0; i < prob->l; i++)
            if (target[i] == prob->y[i])
                ++total_correct;
        printf("Cross Validation Accuracy = %g%%\n",
            100.0 * total_correct / prob->l);
        free(target);
	} else {
		this->model = svm_train(prob, svm_param);
		if (!this->params->modelName.empty()){
			svm_save_model(this->params->modelName.c_str(), this->model);
		}
	}

	free(svm_param);
	for(int i = 0; i < prob->l; i++){
		free(x[i]);
	}
	free(x);
	free(prob);

}



//file output name specified via command line or specially by modifying the parameter struct
void GakcoSVM::write_files() {
	FILE *kernelfile;
	std::string kernelfileName = this->params->outputFilename;
	if(kernelfileName.empty()){
		kernelfileName = "kernel.txt";
	}
	printf("Writing kernel to %s\n", kernelfileName.c_str());
	kernelfile = fopen(kernelfileName.c_str(), "w");
	int nStr = this->nStr;

	for (int i = 0; i < nStr; ++i) {	
		for (int j = 0; j < nStr; ++j) {
			fprintf(kernelfile, "%d:%e ", j + 1, this->kernel[i + j*nStr] );
		}
		fprintf(kernelfile, "\n");
	}
	fclose(kernelfile);
}

void GakcoSVM::write_test_kernel() {
	FILE *kernelfile;
	FILE *labelfile;
	kernelfile = fopen("test_Kernel.txt", "w");
	//labelfile = fopen(this->params->labelFilename.c_str(), "w");
	int nStr = this->nStr;
	int nTestStr = this->nTestStr;
	int num_sv = this->model->nSV[0] + this->model->nSV[1];

	printf("Writing test kernel %d", nTestStr);
	for (int i = 0; i < nTestStr; ++i)
	{	
		for (int j = 0; j < num_sv; ++j)
		{
			fprintf(kernelfile, "%d:%e ", this->model->sv_indices[j], this->test_kernel[j + i*num_sv]);
		}
		//fprintf(labelfile, "%d ", this->labels[i]);
		//fprintf(labelfile, "\n");
		fprintf(kernelfile, "\n");
	}
		
	fclose(kernelfile);
	//fclose(labelfile);
}


double GakcoSVM::predict(double *test_K, int* test_labels){
	int nStr = this->nStr;
	int num_sv = this->model->nSV[0] + this->model->nSV[1];
	int nTestStr = this->nTestStr;
	int totalStr = nStr + nTestStr;
	// test_K = (double*) malloc(totalStr * totalStr * sizeof(double));
	// memset(test_K, 0, totalStr * totalStr * sizeof(double));
	// test_K[0] = 1.0;

	struct svm_node *x = Malloc(struct svm_node, num_sv + 1);
	int correct = 0;
	int pagg =0, nagg=0; //aggregators for finding num of pos and neg samples for auc
	double* neg = Malloc(double, nTestStr);
	double* pos = Malloc(double, nTestStr);
	int labelind = 0;
	for (int i =0; i < 2; i++){
		if (this->model->label[i] == 1)
			labelind = i;
	}
	FILE* labelfile;
	labelfile = fopen(this->params->labelFilename.c_str(), "w");

	for(int i = 0; i < nTestStr; i++){

		for (int j=0; j < num_sv; j++){
			x[j].index = this->model->sv_indices[j];
			x[j].value = test_K[i * num_sv + j];
		}
		x[num_sv].index = -1;

		double probs[2];
		double guess = svm_predict_probability(this->model, x, probs);
		//double guess = svm_predict_values(this->model, x, probs);
		if (test_labels[i] > 0){
			pos[pagg] = probs[labelind];
			pagg += 1;
		} 
		else{
			neg[nagg] = probs[labelind];
			nagg += 1;
		} 

		// if(i <= 2 || i > nTestStr - 4){
		// 	printf("\n%f\n",probs[labelind]);
		// 	printf("Guess %f \t\t Label %d \n", guess, test_labels[i]);
		// }

		fprintf(labelfile, "%d ", (int)guess);
		fprintf(labelfile, "\n");
	
		if ((guess < 0.0 && test_labels[i] < 0) || (guess > 0.0 && test_labels[i] > 0)){
			correct++;
		}
	}

	double auc = calculate_auc(pos, neg, pagg, nagg);
	printf("\nacc: %f\n", (double)correct / nTestStr);

	fclose(labelfile);
	free(pos);
	free(neg);

	return auc;
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
	double* K;
	int lines = 0;
	if (inpfile.is_open()){
		while(!inpfile.eof()){
			std::getline(inpfile, line);
			lines++;
		}
		//int width = find last number for test kernel loading too
		K = (double*) malloc(lines * lines * sizeof(double));
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
	}


	if(this->kernel != NULL)
		free(this->kernel);
	this->kernel = K;
	return K;
}	