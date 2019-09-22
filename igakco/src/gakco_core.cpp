#include "gakco_core.hpp"
#include "shared.h"
#include "libsvm-code/libsvm.h"

#include <thread>
#include <vector>
#include <stdlib.h>
#include <cstdlib>
#include <math.h>
#include <cstring>
#include <algorithm>
#include <random>
#include <ctime>


#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

void kernel_build_parallel(int tid, WorkItem *workQueue, int queueSize,
    pthread_mutex_t *mutexes, kernel_params *params, double *Ksfinal) {

    int itemNum = tid;

    Feature *features = params->features;
    int nfeat = (*features).n;
    int *feat = (*features).features;
    int g = params->g;
    int m = params->m;
    int k = params->k;
    long int n_str_pairs = params->n_str_pairs;
    long int total_str = params->total_str;
    int dict_size = params->dict_size;
    int num_mutex = params->num_mutex;
    int num_threads = params->num_threads;
    bool quiet = params->quiet;

    int num_comb = nchoosek(g, k);

    // where this thread will store its work
    unsigned int *Ks = (unsigned int *) malloc(n_str_pairs * sizeof(unsigned int));
    memset(Ks, 0, sizeof(unsigned int) * n_str_pairs);

    bool working = true;

    while (working) {
        WorkItem workItem = workQueue[itemNum];

        // specifies which partial kernel is to be computed
        int combo_num = workItem.combo_num;
        Combinations *combinations = (Combinations *) malloc(sizeof(combinations));
        (*combinations).n = g;
        (*combinations).k = k;
        (*combinations).num_comb = num_comb;

        // an array of gmer indices associated with group_srt and features_srt
        unsigned int *sortIdx = (unsigned int *) malloc(nfeat * sizeof(unsigned int));
        // sorted gmers
        unsigned int *features_srt = (unsigned int *) malloc(nfeat * g * sizeof(unsigned int));
        // the gmer numbers; associated with features_srt and sortIdx
        unsigned int *group_srt = (unsigned int *) malloc(nfeat * sizeof(unsigned int));
        unsigned int *cnt_comb = (unsigned int *) malloc(2 * sizeof(unsigned int)); //
        // the sorted features once mismatch positions are removed
        unsigned int *feat1 = (unsigned int *) malloc(nfeat * g * sizeof(unsigned int)); 

        int *pos = (int *) malloc(nfeat * sizeof(int));
        memset(pos, 0, sizeof(int) * nfeat);

        unsigned int *out = (unsigned int *) malloc(k * num_comb * sizeof(unsigned int));
        unsigned int *cnt_m = (unsigned int *) malloc(g * sizeof(unsigned int));
        cnt_comb[0] = 0;
        getCombinations((*combinations).n, (*combinations).k, pos, 0, 0, cnt_comb, out, num_comb);
        cnt_m[m] = cnt_comb[0];
        cnt_comb[0] += ((*combinations).k * num_comb);

        // remove mismatch positions
        for (int j1 = 0; j1 < nfeat; ++j1) {
            for (int j2 = 0; j2 < k; ++j2) {
                feat1[j1 + j2 * nfeat] = feat[j1 + (out[(cnt_m[m] - num_comb + combo_num) + j2 * num_comb]) * nfeat];
            }
        }

        // sort the g-mers (this is relatively fast)
        cntsrtna(sortIdx, feat1, k, nfeat, dict_size);

        for (int j1 = 0; j1 < nfeat; ++j1) {
            for (int j2 = 0; j2 <  k; ++j2) {
                features_srt[j1 + j2*nfeat] = feat1[(sortIdx[j1]) + j2*nfeat];
            }
            group_srt[j1] = (*features).group[sortIdx[j1]];
        }

        // update cumulative mismatch profile (slow)
        countAndUpdateTri(Ks, features_srt, group_srt, k, nfeat, total_str);

        free(cnt_m);
        free(out);
        free(sortIdx);
        free(features_srt);
        free(group_srt);
        free(feat1);
        free(cnt_comb);
        free(pos);
        free(combinations);

        // Check if the thread needs to handle more mismatch profiles
        itemNum += num_threads;
        if (itemNum >= queueSize) {
            working = false;
        }
    }

    // set up the mutexes to lock as you go through the matrix
    int cusps[num_mutex];
    for (int i = 0; i < num_mutex; i++) {
        cusps[i] = (int) (i * ((double) n_str_pairs) / num_mutex);
    }

    /* the feared kernel update step, locking is necessary to keep it thread-safe.
    current locking strategy involves splitting the array rows into groups and locking per group.
    also tried going top->bottom or bottom->top dependent on work order to split
    contention among the locks, seemed to split up contention but made it slightly slower?
    */
    int count = 0;
    if (num_threads > 1) {
        for (int j1 = 0; j1 < n_str_pairs; ++j1) {
            if (j1 == cusps[count]) {
                if (count != 0) {
                    pthread_mutex_unlock(&mutexes[count - 1]);
                }
                pthread_mutex_lock(&mutexes[count]);
                if (count + 1 < num_mutex) count++;
            }
            unsigned int val = Ks[j1];
            if (val != 0) Ksfinal[j1] += val;
        }
        pthread_mutex_unlock(&mutexes[num_mutex - 1]);
    } else {
        for (int i = 0; i < n_str_pairs; i++) {
            unsigned int val = Ks[i];
            if (val != 0) Ksfinal[i] += val;
        }
    }
    free(Ks);
}

double* construct_kernel(kernel_params *params) {

    /* Build work queue - represents the partial kernel computations
    that need to be completed by the threads */
    int numCombinations = nchoosek(params->g, params->m);
    
    std::vector<int> indexes(numCombinations);
    for (int i = 0; i < numCombinations; i++) {
        indexes[i] = i;
    }

    numCombinations = params->approx ? int((1 - params->epsilon) * numCombinations) : numCombinations;

    auto rng = std::default_random_engine {};
    rng.seed(std::time(0));
    std::shuffle(std::begin(indexes), std::end(indexes), rng);

    int queueSize = numCombinations;
    WorkItem *workQueue = new WorkItem[queueSize];
    int itemNum = 0;

    for (int i = 0; i < numCombinations; i++) {
        workQueue[i].m = params->m;
        workQueue[i].combo_num = indexes[i];
    }

    // for (int combo_num = 0; combo_num < numCombinations; combo_num++) {
    //     workQueue[combo_num].m = params->m;
    //     workQueue[combo_num].combo_num = combo_num;
    // }

    /* Allocate gapped k-mer kernel */
    double *K = (double *) malloc(params->n_str_pairs * sizeof(double));
    memset(K, 0, params->n_str_pairs * sizeof(double));

    /* Determine how many threads to use */
    int num_threads = params->num_threads;
    if (num_threads == -1) {
        // returns number of processors
        // int numCores = std::thread::hardware_concurrency();
        // num_threads = (numCores > 20) ? 20 : numCores;
        num_threads = 20;
    }
    num_threads = (num_threads > queueSize) ? queueSize : num_threads;

    /* Create an array of mutex locks */
    int num_mutex = params->num_mutex;
    num_mutex = (num_mutex == -1 || num_mutex > num_threads) ? num_threads : num_mutex; 
    pthread_mutex_t *mutexes = (pthread_mutex_t*) malloc(num_mutex * sizeof(pthread_mutex_t));
    for (int i = 0; i < num_mutex; i++){
        pthread_mutex_init(&mutexes[i], NULL);
    }

    params->num_threads = num_threads;
    params->num_mutex = num_mutex;

    /* Multithreaded kernel construction */
    if (!params->quiet) printf("Computing %d mismatch profiles using %d threads...\n", numCombinations, num_threads);
    std::vector<std::thread> threads;
    for (int tid = 0; tid < num_threads; tid++) {
        threads.push_back(std::thread(kernel_build_parallel, tid, workQueue, queueSize, mutexes, params, K));
    }

    for (auto &t : threads) {
        t.join();
    }

    /* Kernel normalization */

    for (int i = 0; i < params->total_str; i++) {
        for (int j = 0; j < i; j++) {
            tri_access(K, i, j) = tri_access(K, i, j) / sqrt(tri_access(K, i, i) * tri_access(K, j, j));
        }
    }
    for (int i = 0; i < params->total_str; i++) {
        tri_access(K, i, i) = tri_access(K, i, i) / sqrt(tri_access(K, i, i) * tri_access(K, i, i));
    }

    return K;
}

double *construct_test_kernel(int n_str_train, int n_str_test, double *K) {
    double* test_K = (double*) malloc(n_str_test * n_str_train * sizeof(double));
    int total_str = n_str_train + n_str_test;
    for(int i = n_str_train; i < total_str; i++){
        for(int j = 0; j < n_str_train; j++){
            test_K[(i - n_str_train) * n_str_train + j] 
                = tri_access(K, i, j) / sqrt(tri_access(K, i, i) * tri_access(K, j, j));
        }
    }
    return test_K;
}

double *run_cross_validation(double *K, std::string metric, int k) {
    return 0;
}

svm_problem *create_svm_problem(double *K, int *labels, kernel_params *kernel_param, svm_parameter *svm_param) {
    long int n_str = kernel_param->total_str;
    long int n_str_train = kernel_param->n_str_train;

    struct svm_problem* prob = Malloc(svm_problem, 1);
    const char* error_msg;

    svm_node** x;
    svm_node* x_space;
    prob->l = n_str_train;
    prob->y = Malloc(double, prob->l);
    x = Malloc(svm_node*, prob->l);
    if (svm_param->kernel_type == GAKCO) {
        x_space = Malloc(struct svm_node, (n_str_train + 1) * n_str_train);
        int totalind = 0;
        for (int i = 0; i < n_str_train; i++) {
            x[i] = &x_space[totalind];
            for (int j = 0; j < n_str_train; j++) {
                x_space[j + i * (n_str_train + 1)].index = j + 1; 
                x_space[j + i * (n_str_train + 1)].value = tri_access(K, i, j);
            }
            totalind += n_str_train;
            x_space[totalind].index = -1;
            totalind++;
            prob->y[i] = labels[i];
        }
        //this->x_space = x_space;
    } else if (svm_param->kernel_type == LINEAR || svm_param->kernel_type == RBF) {
        x_space = Malloc(struct svm_node, (n_str_train + 1) * n_str_train);
        int totalind = 0;
        for (int i = 0; i < n_str_train; i++) {
            x[i] = &x_space[totalind];
            for (int j = 0; j < n_str_train; j++) {
                x_space[j + i * (n_str_train + 1)].index = j + 1; 
                x_space[j + i * (n_str_train + 1)].value = tri_access(K, i, j);
            }
            totalind += n_str_train;
            x_space[totalind].index = -1;
            totalind++;
            prob->y[i] = labels[i];
        }
        //this->x_space = x_space;
    }

    prob->x = x;

    // if quiet mode, set libsvm's print function to null
    if (kernel_param->quiet) {
        svm_set_print_string_function(&print_null);
    }

    error_msg = svm_check_parameter(prob, svm_param);

    if (error_msg) {
        fprintf(stderr, "ERROR: %s\n", error_msg);
        exit(1);
    }

    return prob;
}

svm_model *train_model(double *K, int *labels, kernel_params *kernel_param, svm_parameter *svm_param) {
    long int n_str = kernel_param->total_str;
    long int n_str_train = kernel_param->n_str_train;
    struct svm_problem* prob = Malloc(svm_problem, 1);
    const char* error_msg;

    svm_node** x;
    svm_node* x_space;
    prob->l = n_str_train;
    prob->y = Malloc(double, prob->l);
    x = Malloc(svm_node*, prob->l);
    if (svm_param->kernel_type == GAKCO) {
        x_space = Malloc(struct svm_node, (n_str_train + 1) * n_str_train);
        int totalind = 0;
        for (int i = 0; i < n_str_train; i++) {
            x[i] = &x_space[totalind];
            for (int j = 0; j < n_str_train; j++) {
                x_space[j + i * (n_str_train + 1)].index = j + 1; 
                x_space[j + i * (n_str_train + 1)].value = tri_access(K, i, j);
            }
            totalind += n_str_train;
            x_space[totalind].index = -1;
            totalind++;
            prob->y[i] = labels[i];
        }
        //this->x_space = x_space;
    } else if (svm_param->kernel_type == LINEAR || svm_param->kernel_type == RBF) {
        x_space = Malloc(struct svm_node, (n_str_train + 1) * n_str_train);
        int totalind = 0;
        for (int i = 0; i < n_str_train; i++) {
            x[i] = &x_space[totalind];
            for (int j = 0; j < n_str_train; j++) {
                x_space[j + i * (n_str_train + 1)].index = j + 1; 
                x_space[j + i * (n_str_train + 1)].value = tri_access(K, i, j);
            }
            totalind += n_str_train;
            x_space[totalind].index = -1;
            totalind++;
            prob->y[i] = labels[i];
        }
        //this->x_space = x_space;
    }

    prob->x = x;

    // if quiet mode, set libsvm's print function to null
    if (kernel_param->quiet) {
        svm_set_print_string_function(&print_null);
    }

    error_msg = svm_check_parameter(prob, svm_param);

    if (error_msg) {
        fprintf(stderr, "ERROR: %s\n", error_msg);
        exit(1);
    }

    // train that ish
    struct svm_model* model;
    model = svm_train(prob, svm_param);

    free(svm_param);

    return model;
}
