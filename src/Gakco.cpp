// GaKCo : Fast Gapped k-mer string Kernel using Counting
// Code Contibution by:
//Ritambhara Singh <rs3zz@virginia.edu>
//Kamran Kowsari <kk7nc@virginia.edu >
//Arshdeep Sekhon <as5cu@virginia.edu >



// This file contains Main Code


#include <stdio.h>
#include <cstdlib>
#include <string.h>
#include <math.h>
//#include "shared.h"
//#include "shared.cpp"
#include <assert.h>
#include <thread>
#include <iostream>
#include <random>
#include <ctime>
#include <fstream>
#include "readInput.h"
#include <future>
#include <mutex>
#include <pthread.h>
#include <unistd.h>
#include "GakcoSVM.h"
#include "Gakco.h"



int help() {
	printf("\nUsage: gakco [options] <trainingFile> <testingFile> <dictionaryFile> <labelsFile>\n");
	printf("FLAGS WITH ARGUMENTS\n");
	printf("\t g : gmer length; length of substrings (allowing up to m mismatches) used to compare sequences. Constraints: 0 < g < 20\n");
	printf("\t m : maximum number of mismatches when comparing two gmers. Constraints: 0 <= m < g\n");
	printf("\t t : (optional) number of threads to use. Set to 1 to not parallelize kernel computation\n");
	printf("\t C : (optional) SVM C parameter. Default is 1.0\n");
	printf("\t k : (optional) Specify a kernel filename to print to. If -l is also set, this will instead be used as the filename to load the kernel from\n");
	printf("\t o : (optional) Specify a model filename to print to. If -s is also set, this will instead be used as the filename to load the model from\n");
	printf("\t r : (optional) 1 for GAKCO (default), 2 for LINEAR");
	printf("\t S : (optional) Specifies number of mutexes to use during kernel update. Default: 1");
	printf("NO ARGUMENT FLAGS\n");
	printf("\t l : (optional) If set, will load the train kernel from the file specified by -k\n");
    printf("\t s : (optional) If set, will load the train kernel from the file specified by -k and will load the model from the file specified by -o\n");
    printf("\t p : (optional) Flag for model to generate probability of class. Without it, AUC can't be calculated.\n");
    printf("\t h : (optional) set to 1 or 2. If 1, will halt the program after constructing and printing out the kernel. If 2, will halt after training and printing out the model\n");
	printf("ORDERED PARAMETERS\n");
	printf("\t trainingFile : set of training examples in FASTA format\n");
	printf("\t testingFile : set of testing examples in FASTA format\n");
	printf("\t dictionaryFile : file containing the alphabet of characters that appear in the sequences (simple text file)\n");
	printf("\t labelsFile : file to place labels from the test examples (simple text file)\n");
	printf("\n");
	printf("\nExample usage: ./iGakco -g 7 -m 2 -t 4 -C 1.0 trainingSet.fasta testingSet.fasta proteinDictionary.txt labelOutputFile.txt\n\n");
	return 1;
}

int errorID1() {
	printf("Error: g >= Shortest sequence in the input file!\n");
	return 1;
}

//builder for the train kernel, triangularized to save memory
//Accepts unsigned int** Ksfinal instead of unsigned int*
void build_cumulative_mismatch_profiles_tri(WorkItem *workQueue, int queueSize, int threadNum, int numThreads, int *elems, 
										Features *features, double *Ksfinal, int *feat, int g, int na,
										int nfeat, int nStr, pthread_mutex_t *mutex, int num_mutex, int quiet) {
	bool working = true;
	int itemNum = threadNum;
	// WorkItem* threadQueue = new WorkItem[(queueSize / numThreads)+1];
	// int i = 0;
	// while(itemNum < queueSize){
	// 	threadQueue[i] = workQueue[itemNum];
	// 	itemNum += numThreads;
	// 	i++;
	// }
	// //shuffle the workItems for this thread after splitting, so each thread has an equal distribution of different m's 
	// //but performs them in different order so as to avoid bottlenecking on the sequential accumulation into C_m
	// shuffle(threadQueue, i);
	// i--;

	int num_str_pairs = nStr * (nStr+1) / 2;
	while (working) {
		//Determine which mismatch profile needs to be computed by this thread
		//WorkItem workItem = threadQueue[i];
		WorkItem workItem = workQueue[itemNum];
		int m = workItem.m;
		int combo_num = workItem.combo_num; //specifies which mismatch profile for the given value of m is to be computed

		int k = g - m;
		int num_comb = nchoosek(g, k); //Number of possible mismatch positions
		Combinations * combinations = (Combinations *) malloc(sizeof(Combinations));
		(*combinations).n = g;
		(*combinations).k = k;
		(*combinations).num_comb = num_comb;
		
		unsigned int *Ks = (unsigned int *) malloc(num_str_pairs * sizeof(unsigned int)); //where this thread will store its work
		unsigned int *sortIdx = (unsigned int *) malloc(nfeat * sizeof(unsigned int)); //an array of gmer indices associated with group_srt and features_srt
		unsigned int *features_srt = (unsigned int *) malloc(nfeat * g * sizeof(unsigned int)); //sorted features
		unsigned int *group_srt = (unsigned int *) malloc(nfeat * sizeof(unsigned int)); //the gmer numbers; associated with features_srt and sortIdx
		unsigned int *cnt_comb = (unsigned int *) malloc(2 * sizeof(unsigned int)); //
		unsigned int *feat1 = (unsigned int *) malloc(nfeat * g * sizeof(unsigned int)); //the sorted features once mismatch positions are removed
		
		int *pos = (int *) malloc(nfeat * sizeof(int));
		memset(pos, 0, sizeof(int) * nfeat);

		unsigned int *out = (unsigned int *) malloc(k * num_comb * sizeof(unsigned int));
		unsigned int *cnt_m = (unsigned int *) malloc(g * sizeof(unsigned int));
		cnt_comb[0] = 0;
		getCombinations(elems, (*combinations).n, (*combinations).k, pos, 0, 0, cnt_comb, out, num_comb);
		cnt_m[m] = cnt_comb[0];
		cnt_comb[0] += ((*combinations).k * num_comb);
		
		//remove i positions
		for (int j1 = 0; j1 < nfeat; ++j1) {
			for (int j2 = 0; j2 < k; ++j2) {
				feat1[j1 + j2 * nfeat] = feat[j1 + (out[(cnt_m[m] - num_comb + combo_num) + j2 * num_comb]) * nfeat];
			}
		}

		//sort the g-mers (this is relatively fast)
		cntsrtna(sortIdx, feat1, k, nfeat, na);    

		for (int j1 = 0; j1 < nfeat; ++j1) {
			for (int j2 = 0; j2 <  k; ++j2) {
				features_srt[j1 + j2*nfeat] = feat1[(sortIdx[j1]) + j2*nfeat];
			}
			group_srt[j1] = (*features).group[sortIdx[j1]];
		}
		//update cumulative mismatch profile (slow)
		countAndUpdateTri(Ks, features_srt, group_srt, k, nfeat, nStr);

		//set up the mutexes to lock as you go through the matrix
		int cusps[num_mutex];
		for (int i = 0; i < num_mutex; i++){
			cusps[i] = (int)((i)*((double)nStr)/num_mutex);
		}

		//the feared kernel update step, locking is necessary to keep it thread-safe
		//current locking strategy involves splitting the array rows into groups and locking per group
		//also tried going top->bottom or bottom->top dependent on work order to split contention among the locks, seemed to split up contention but made it slightly slower?
		int count = 0;
		for (int j1 = 0; j1 < nStr; ++j1) {
			if (j1 ==cusps[count]){
				if (count != 0)
					pthread_mutex_unlock(&mutex[count-1]);
				pthread_mutex_lock(&mutex[count]);
				count++;
			}
			for (int j2 = j1; j2 < nStr; ++j2) {
				tri_access(Ksfinal, j1, j2) += tri_access(Ks, j1, j2);
			}
		}
		pthread_mutex_unlock(&mutex[num_mutex-1]);
		

		free(cnt_m);
		free(out);
		free(Ks);
		free(sortIdx);
		free(features_srt);
		free(group_srt);
		free(feat1);
		free(cnt_comb);
		free(pos);
		free(combinations);

		//Check if the thread needs to handle more mismatch profiles
		itemNum += numThreads;
		if (itemNum >= queueSize) {
			working = false;
			if(!quiet)
				printf("Thread %d finished...\n", threadNum);
		}
	}
}


double igakco_main_wrapper(int argc, char* argv[]){
	// Get g, k, nStr, and t values from command line
	struct gakco_param arg;
	int g = -1;
	int M = -1;
	int numThreads = -1;
	int probability = 0;
	float C = -1;
	char c;
	int onlyprint = 0;
	int nopredict = 0;
	int quiet= 0;

  	optind =0;
	while ((c = getopt(argc, argv, "g:m:t:C:k:o:h:r:lspq")) != -1) {
		switch (c) {
			case 'g':
				g = atoi(optarg);
				break;
			case 'm':
				M = atoi(optarg);
				break;
			case 't':
				numThreads = atoi(optarg);
				break;
			case 'C':
				C = atof(optarg);
				break;
			case 'p':
				probability = 1;
				break;
			case 'k':
				arg.outputFilename = optarg;
				break;
			case 'o':
				arg.modelName = optarg;
				break;
			case 'l':
				arg.loadkernel = 1;
				break;
			case 's':
				arg.loadkernel = 1;
				arg.loadmodel = 1;
				 break;
			case 'r':
				if (atoi(optarg) == 1)
					arg.kernel_type = LINEAR;
				else
					arg.kernel_type = GAKCO;
				break;
			case 'h':
				if (atoi(optarg) == 1){
					onlyprint = 1;
				}else if (atoi(optarg) == 2){
					nopredict = 1;
				}
				break;
			case 'q':
				quiet = 1;
				break;

        break;
		}
	}
	if (g == -1) {
		printf("Must provide a value for the g parameter\n");
		return help();
	}
	if (M == -1) {
		printf("Must provide a value for the m parameter\n");
		return help();
	}


	int argNum = optind;
	// Get names of sequence, dictionary, labels, and kernel files 	
	char filename[100], testFilename[100], filename_label[100], Dicfilename[100];
	strcpy(filename, argv[argNum++]);
	strcpy(testFilename, argv[argNum++]);
	strcpy(Dicfilename, argv[argNum++]);
	strcpy(filename_label, argv[argNum++]);

	char isVerbose;
	isVerbose = 0;
	
	// Create a Gakco SVM object to construct kernel, train, and test
	arg.filename = filename;
	arg.testFilename = testFilename;
	arg.dictFilename = Dicfilename;
	arg.labelFilename = filename_label;
	arg.g = g;
	arg.k = g - M;
	arg.probability = probability;
	arg.quiet = quiet;
	if (numThreads != -1) {
		arg.threads = numThreads;
	}
	if (C != -1) {
		arg.C = C;
	}
	arg.eps = .001;
	arg.h = 0;
	

	//Create GakcoSVM object with specified params. Params can be modified in between kernel construction
	//or training to make changes to how it behaves next time.
	GakcoSVM gsvm = GakcoSVM(&arg);

	if(arg.loadkernel)
		gsvm.load_kernel(arg.outputFilename);

	//returns a pointer to the kernel matrix, and also stores it as a member variable
	//still needed even if we are loading a kernel as it reads labels and dataset info, but won't calculate if it doesn't need to.
	if(arg.kernel_type == GAKCO)
		gsvm.construct_kernel();
	else if(arg.kernel_type == LINEAR)
		gsvm.construct_linear_kernel();

	if(!arg.loadkernel && !arg.outputFilename.empty()){
		gsvm.write_files();
		//gsvm.write_libsvm_kernel();
	}

	//exit program if only need to print out the kernel
	if (onlyprint)
		return 0;

	if(arg.loadmodel){
		gsvm.model = svm_load_model(arg.modelName.c_str());
	}else{
		//trains the SVM on the provided dataset, outputs a model into modelName if provided
		gsvm.train(gsvm.kernel);
	}
	
	if(nopredict)
		return 0;
	if(arg.kernel_type == GAKCO)
		gsvm.construct_test_kernel();

	gsvm.write_test_kernel();

	double acc = gsvm.predict(gsvm.test_kernel, gsvm.test_labels);


	return acc;
}

//Main function 
int main(int argc, char *argv[]) {
		// Get g, k, nStr, and t values from command line
	struct gakco_param arg;
	int g = -1;
	int M = -1;
	int numThreads = -1;
	int probability = 0;
	float C = -1;
	int c;
	int onlyprint = 0;
	int nopredict = 0;
	int quiet= 0;
  
	while ((c = getopt(argc, argv, "g:m:t:C:k:o:h:r:S:lspq")) != -1) {
		switch (c) {
			case 'g':
				g = atoi(optarg);
				break;
			case 'm':
				M = atoi(optarg);
				break;
			case 't':
				numThreads = atoi(optarg);
				break;
			case 'C':
				C = atof(optarg);
				break;
			case 'p':
				probability = 1;
				break;
			case 'k':
				arg.outputFilename = optarg;
				break;
			case 'o':
				arg.modelName = optarg;
				break;
			case 'l':
				arg.loadkernel = 1;
				break;
			case 's':
				arg.loadkernel = 1;
				arg.loadmodel = 1;
				break;
			case 'S':
				arg.num_mutexes = atoi(optarg);
				break;
			case 'r':
				if (atoi(optarg) == 1)
					arg.kernel_type = LINEAR;
				else
					arg.kernel_type = GAKCO;
				break;
			case 'h':
				if (atoi(optarg) == 1){
					onlyprint = 1;
				}else if (atoi(optarg) == 2){
					nopredict = 1;
				}
				break;
			case 'q':
				quiet = 1;
				break;

        break;
		}
	}
	if (g == -1) {
		printf("Must provide a value for the g parameter\n");
		return help();
	}
	if (M == -1) {
		printf("Must provide a value for the m parameter\n");
		return help();
	}


	int argNum = optind;
	// Get names of sequence, dictionary, labels, and kernel files 	
	char filename[100], testFilename[100], filename_label[100], Dicfilename[100];
	strcpy(filename, argv[argNum++]);
	strcpy(testFilename, argv[argNum++]);
	strcpy(Dicfilename, argv[argNum++]);
	strcpy(filename_label, argv[argNum++]);

	char isVerbose;
	isVerbose = 0;
	
	// Create a Gakco SVM object to construct kernel, train, and test
	arg.filename = filename;
	arg.testFilename = testFilename;
	arg.dictFilename = Dicfilename;
	arg.labelFilename = filename_label;
	arg.g = g;
	arg.k = g - M;
	arg.probability = probability;
	arg.quiet = quiet;
	if (numThreads != -1) {
		arg.threads = numThreads;
	}
	if (C != -1) {
		arg.C = C;
	}
	arg.eps = .001;
	arg.h = 0;
	

	//Create GakcoSVM object with specified params. Params can be modified in between kernel construction
	//or training to make changes to how it behaves next time.
	GakcoSVM gsvm = GakcoSVM(&arg);

	if(arg.loadkernel)
		gsvm.load_kernel(arg.outputFilename);

	//returns a pointer to the kernel matrix, and also stores it as a member variable
	//still needed even if we are loading a kernel as it reads labels and dataset info, but won't calculate if it doesn't need to.
	if(arg.kernel_type == GAKCO)
		gsvm.construct_kernel();
	else if(arg.kernel_type == LINEAR)
		gsvm.construct_linear_kernel();

	if(!arg.loadkernel && !arg.outputFilename.empty()){
		gsvm.write_files();
		//gsvm.write_libsvm_kernel();
	}

	//exit program if only need to print out the kernel
	if (onlyprint)
		return 0;

	if(arg.loadmodel){
		gsvm.model = svm_load_model(arg.modelName.c_str());
	}else{
		//trains the SVM on the provided dataset, outputs a model into modelName if provided
		gsvm.train(gsvm.kernel);
	}
	
	if(nopredict)
		return 0;
	if(arg.kernel_type == GAKCO)
		gsvm.construct_test_kernel();

	gsvm.write_test_kernel();

	gsvm.predict(gsvm.test_kernel, gsvm.test_labels);


	return 0;
}
