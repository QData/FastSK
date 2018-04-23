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
#include "shared.h"
#include "shared.cpp"
#include <assert.h>
#include <omp.h>
#include <thread>
#include <iostream>
#include <random>
#include <ctime>
#include <fstream>
#include "readInput.cpp"
#include <future>
#include <mutex>
#include <pthread.h>
#include <unistd.h>
#include "GakcoSVM.h"
#include "GaKCo.h"


int help() {
	printf("\nUsage: gakco [options] <trainingFile> <testingFile> <dictionaryFile> <labelsFile> <kernelFile>\n");
	printf("\t g : length of gapped instance. Constraints: 0 < g < 20\n");
	printf("\t k : length of k-mer. Constraints: k < g\n");
	printf("\t n : (optional) maximum number of examples in the data set. Default: 15000\n");
	printf("\t t : (optional) number of threads to use. Set to 1 to not parallelize kernel computation\n");
	printf("\t C : (optional) SVM C parameter. Default is 1.0");
	printf("\t trainingFile : set of training examples in FASTA format\n");
	printf("\t testingFile : set of testing examples in FASTA format\n");
	printf("\t dictionaryFile : file containing the alphabet of characters that appear in the sequences (simple text file)\n");
	printf("\t labelsFile : file to place labels from the examples (simple text file)\n");
	printf("\t kernelFile : name of the file to write the kernel that will be computed by GaKCo\n");
	printf("\n");
	printf("\t IMPORTANT: \n");
	printf("\t\t sequence elements must be in the range [0,AlphabetSize - 1].\n");
	printf("\t\t g - k should be less than 20\n");
	printf("\nExample usage: ./GaKCo -g 7 -k 5 -n 15000 -t 4 -C 1.0 trainingSet.fasta testingSet.fasta proteinDictionary.txt labelOutputFile.txt kernelOutputFile.txt\n\n");

	return 1;
}

int errorID1() {
	printf("Error: g >= Shortest sequence in the input file!\n");
	return 1;
}

void build_cumulative_mismatch_profiles(WorkItem *workQueue, int queueSize, int threadNum, int numThreads, int *elems, 
										Features *features, unsigned int *Ksfinal, int *cnt_k, int *feat, int g, int na,
										int nfeat, int nStr, pthread_mutex_t *mutexes) {
	bool working = true;
	int itemNum = threadNum;
	while (working) {
		//Determine which mismatch profile needs to be computed by this thread
		WorkItem workItem = workQueue[itemNum];
		int m = workItem.m;
		int combo_num = workItem.combo_num; //specifies which mismatch profile for the given value of m is to be computed

		int k = g - m;
		int num_comb = nchoosek(g, k); //Number of possible mismatch positions
		Combinations * combinations = (Combinations *) malloc(sizeof(Combinations));
		(*combinations).n = g;
		(*combinations).k = k;
		(*combinations).num_comb = num_comb;
		unsigned long int c = m * (nStr * nStr);
		
		unsigned int *Ks = (unsigned int *) malloc(nStr*nStr * sizeof(unsigned int)); //where this thread will store its work
		unsigned int *sortIdx = (unsigned int *) malloc(nfeat * sizeof(unsigned int)); //an array of gmer indices associated with group_srt and features_srt
		unsigned int *features_srt = (unsigned int *) malloc(nfeat * g * sizeof(unsigned int *)); //sorted features
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
		countAndUpdate(Ks, features_srt, group_srt, k, nfeat, nStr);

		pthread_mutex_lock(&mutexes[m]);
		for (int j1 = 0; j1 < nStr; ++j1) {
			for (int j2 = j1; j2 < nStr; ++j2) {
				if(j1 != j2) {
					Ksfinal[(c + j1) + j2 * nStr] += Ks[j1 + j2*nStr]; 
				}
				Ksfinal[c + j1 * nStr + j2] += Ks[j1 + j2*nStr];
			}
		}
		pthread_mutex_unlock(&mutexes[m]);

		free(cnt_m);
		free(out);
		cnt_k[m] = c;
		free(Ks);
		free(sortIdx);
		free(features_srt);
		free(group_srt);
		free(feat1);
		free(cnt_comb);
		free(pos);

		//Check if the thread needs to handle more mismatch profiles
		itemNum += numThreads;
		if (itemNum >= queueSize) {
			working = false;
			printf("Thread %d finished...\n", threadNum);
		}
	}
}

void main_test_kernel(int *elems, Features *features, unsigned int *Ksfinal, int *cnt_k, int *feat, 
					int g, int k, int na, int nfeat, int nStr, int nTestStr, int i) {
	unsigned long int c = 0;
	int num_comb;
	Combinations * combinations = (Combinations *)malloc(sizeof(Combinations));
    unsigned int *Ks = (unsigned int *)malloc(nTestStr * nStr * sizeof(unsigned int));
	unsigned int *sortIdx = (unsigned int *)malloc(nfeat * sizeof(unsigned int));
	unsigned int *features_srt  = (unsigned int *)malloc(nfeat*g * sizeof(unsigned int *));
	unsigned int *group_srt = (unsigned int *)malloc(nfeat * sizeof(unsigned int));
	unsigned int *cnt_comb = (unsigned int *)malloc(2 * sizeof(unsigned int));
	unsigned int *feat1 = (unsigned int *)malloc(nfeat*g * sizeof(unsigned int));
	int *pos = (int *)malloc(nfeat * sizeof(int));
	memset(pos, 0, sizeof(int) * nfeat);
	c =i*(nStr * nTestStr);

	(*combinations).n = g;
	(*combinations).k = g - i;
	(*combinations).num_comb = nchoosek(g, g - i);

	// number of possible positions
	num_comb = nchoosek(g, g - i);

	unsigned int  *out = (unsigned int *)malloc((g - i)*num_comb * sizeof(unsigned int));
	unsigned int  *cnt_m = (unsigned int *)malloc(g * sizeof(unsigned int));

	cnt_comb[0] = 0;

	getCombinations(elems,(*combinations).n, (*combinations).k, pos, 0, 0, cnt_comb, out, num_comb);
	cnt_m[i] = cnt_comb[0];

	cnt_comb[0] += ((*combinations).k*num_comb);

	for ( int j = 0; j < num_comb; ++j) {
		//remove i positions
		for ( int j1 = 0; j1 < nfeat; ++j1) {
			for ( int j2 = 0; j2 < g - i; ++j2) {
				feat1[j1 + j2*nfeat] = feat[j1 + (out[(cnt_m[i] - num_comb + j) + j2*num_comb])*nfeat];
			}
		}
		//sort the g-mers
		cntsrtna(sortIdx,feat1, g - i, nfeat, na);
		for ( int j1 = 0; j1 < nfeat; ++j1) {
			for ( int j2 = 0; j2 < g - i; ++j2) {
				features_srt[j1 + j2*nfeat] = feat1[(sortIdx[j1]) + j2*nfeat];
			}
			group_srt[j1] = (*features).group[sortIdx[j1]];
		}
		//update cumulative mismatch profile
		countAndUpdateTest(Ks, features_srt, group_srt, g - i, nfeat, nStr, nTestStr);
		for ( int j1 = 0; j1 < nStr; ++j1) {
			for ( int j2 = j1; j2 < nStr; ++j2) {
				if (j1 != j2) {
					Ksfinal[(c + j1) + j2*nStr] +=  Ks[j1 + j2*nStr];
				}
				Ksfinal[c +(j1)*nStr + j2] +=  Ks[j1 + j2*nStr];
			}
		}		
	}
}

Features* merge_features(Features* train, Features* test, int g) {
	int nfeat = train->n + test->n;
	int* features = (int *)malloc((nfeat)*g * sizeof(int *));
	int* group = (int *)malloc(nfeat * sizeof(int));

	memcpy(features, train->features, train->n * g * sizeof(int*));
	memcpy(&(features[train->n * g]), test->features, test->n * g * sizeof(int*));

	memcpy(group, train->group, train->n * sizeof(int));
	memcpy(&(group[train->n]), test->group, test->n * sizeof(int));

	Features* F = (Features *)malloc(sizeof(Features));
	(*F).features = features;
	(*F).group = group;
	(*F).n = nfeat;

	//free(test);

	return F;
}

//Main function 

int main(int argc, char *argv[]) {
	// Get g, k, nStr, and t values from command line
	int g = -1;
	int k = -1;
	int numThreads = -1;
	double C = -1;

	int c;
	while ((c = getopt(argc, argv, "g:k:n:t:C")) != -1) {
		switch (c) {
			case 'g':
				g = atoi(optarg);
				break;
			case 'k':
				k = atoi(optarg);
				break;
			case 't':
				numThreads = atoi(optarg);
				break;
			case 'C':
				C = atof(optarg);
				break;
		}
	}
	if (g == -1) {
		printf("Must provide a value for the g parameter\n");
		return help();
	}
	if (k == -1) {
		printf("Must provide a value for the k parameter\n");
		return help();
	}

	int argNum = optind;
	// Get names of sequence, dictionary, labels, and kernel files 	
	char filename[100], testFilename[100], filename_label[100], Dicfilename[100], opfilename[100];
	strcpy(filename, argv[argNum++]);
	strcpy(testFilename, argv[argNum++]);
	strcpy(Dicfilename, argv[argNum++]);
	strcpy(filename_label, argv[argNum++]);
	strcpy(opfilename, argv[argNum++]);

	int *label;
	long int nStr;
	double *K, *test_K;
	char isVerbose;
	isVerbose = 0;
	
	// Create a Gakco SVM object to construct kernel, train, and test
	struct gakco_param arg;
	arg.filename = filename;
	arg.testFilename = testFilename;
	arg.dictFilename = Dicfilename;
	arg.labelFilename = filename_label;
	arg.outputFilename = opfilename;
	arg.g = g;
	arg.k = k;
	if (numThreads != -1) {
		arg.threads = numThreads;
	}
	if (C != -1) {
		arg.C = C;
	}
	//arg.crossfold = 1;
	//arg.h = 0;
	arg.eps = .001;
	arg.h = 0;
	arg.kernel_type = GAKCO;
	arg.probability = 1;

	//Create GakcoSVM object with specified params. Params can be modified in between kernel construction
	//or training to make changes to how it behaves next time.
	GakcoSVM gsvm = GakcoSVM(&arg);

	//returns a pointer to the kernel matrix, and also stores it as a member variable
	K = gsvm.construct_kernel();
	//gsvm.write_files();

	//K = gsvm.load_kernel(std::string("../release/src/kernel.txt"), std::string("labels.txt"));

	//trains the SVM on the provided dataset, outputs a model into GaKCoModel.txt
	gsvm.train(K);
	//gsvm.write_files();

	

	test_K = gsvm.construct_test_kernel();
	//gsvm.write_test_kernel();

	double acc = gsvm.predict(test_K, gsvm.test_labels);

	//FILE* outfile = fopen("rezzies.txt", "a");

	//fprintf(outfile, "\n%s\n%f\n", filename, acc); 
	//fclose(outfile);

	printf("\nauc: %f\n", acc);


	return 0;
}
