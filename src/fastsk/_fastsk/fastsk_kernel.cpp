#include "fastsk_kernel.hpp"
#include "shared.h"
#include <thread>
#include <vector>
#include <stdlib.h>
#include <cstdlib>
#include <math.h>
#include <cstring>
#include <algorithm>
#include <random>
#include <ctime>
#include <cmath>
#include <string>
#include <iostream>
#include <fstream>

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

KernelFunction::KernelFunction(kernel_params* params) {
    std::cout << "Initializing kernel function" << std::endl;
    this->params = params;
}

double* KernelFunction::compute_kernel() {
    kernel_params* params = this->params;

    /* Build work queue - represents the partial kernel computations
    that need to be completed by the threads */
    int numCombinations = nchoosek(params->g, params->m);
    
    std::vector<int> indexes(numCombinations);
    for (int i = 0; i < numCombinations; i++) {
        indexes[i] = i;
    }

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
    for (int i = 0; i < num_mutex; i++) {
        pthread_mutex_init(&mutexes[i], NULL);
    }

    params->num_threads = num_threads;
    params->num_mutex = num_mutex;

    // If central theorem unlikely to apply, compute exact kernel
    // if (numCombinations / num_threads < 50) {
    //     params->approx = false;
    // }
    if (params->approx) {
        printf("Computing approximate kernel...\n");
    } else {
        printf("Computing exact kernel...\n");
    }

    /* Multithreaded kernel construction */
    if (!params->quiet) printf("Computing %d mismatch profiles using %d threads...\n", numCombinations, num_threads);
    std::vector<std::thread> threads;
    for (int tid = 0; tid < num_threads; tid++) {
        threads.push_back(std::thread(&KernelFunction::kernel_build_parallel, this, tid, workQueue, queueSize, mutexes, params, K));
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

double KernelFunction::get_variance(unsigned int *Ks, double *K_hat, double *variances, int n_str_pairs, int n_train_pairs, int iter) {
    double max_variance = 0;
    double avg_variance = 0;
    int count = 0;
    double delta;
    double delta2;
    double product;
    
    for (int i = 0; i < n_str_pairs; i++) {
        delta = Ks[i] - K_hat[i];
        K_hat[i] += delta / iter;
        
        if (i < n_train_pairs) {
            delta2 = Ks[i] - K_hat[i];
            product = delta * delta2;
            variances[i] += product;
            avg_variance += product;
            // variances[i] += std::pow(delta, 2.0) / iter + math.pow(variances[i], 2.0) / (iter - 1);
            if (variances[i] > max_variance) {
                max_variance = variances[i];
            }
            count++;
        }
    }

    avg_variance /= count;
    if (iter == 1) {
        avg_variance = 9999999;
        max_variance = 9999999;
    } else {
        avg_variance /= iter - 1;
        max_variance /= max_variance / (iter - 1);
    }

    return avg_variance;
}

void KernelFunction::kernel_build_parallel(int tid, WorkItem *workQueue, int queueSize,
    pthread_mutex_t *mutexes, kernel_params *params, double *Ksfinal) {

    int itemNum = tid;
    Feature *features = params->features;
    int nfeat = (*features).n;
    int *feat = (*features).features;
    int g = params->g;
    int m = params->m;
    int k = params->k;
    int n_str_train = params->n_str_train;
    int n_str_test = params->n_str_test;
    long int n_str_pairs = params->n_str_pairs;
    long int total_str = params->total_str;
    int dict_size = params->dict_size;
    int num_mutex = params->num_mutex;
    int num_threads = params->num_threads;
    double delta = params->delta;
    bool quiet = params->quiet;
    bool approx = params->approx;
    int max_iters = params->max_iters;
    bool skip_variance = params->skip_variance;

    int num_comb = nchoosek(g, k);
    long int n_train_pairs = (n_str_train / (double) 2) * (n_str_train + 1);
    long int n_test_pairs = (n_str_test / (double) 2) * (n_str_test + 1);

    bool working = true;
    int iter = 1;

    unsigned int* Ks = (unsigned int*) malloc(sizeof(unsigned int) * n_str_pairs);
    memset(Ks, 0, sizeof(unsigned int) * n_str_pairs);
    
    double* K_hat;
    double* variances;

    if (approx && !skip_variance) {
        K_hat = (double*) malloc(sizeof(double) * n_str_pairs);
        variances = (double*) malloc(sizeof(double) * n_train_pairs);
        memset(K_hat, 0, sizeof(double) * n_str_pairs);
        memset(variances, 0, sizeof(double) * n_train_pairs);
    }

    while (working) {
        WorkItem workItem = workQueue[itemNum];

        // don't cumulate mismatch profiles if computing partial kernel variances
        if (approx && !skip_variance) {
            memset(Ks, 0, sizeof(unsigned int) * n_str_pairs);
        }

        // specifies which partial kernel is to be computed
        int combo_num = workItem.combo_num;
        Combinations *combinations = (Combinations *) malloc(sizeof(combinations));
        (*combinations).n = g;
        (*combinations).k = k;
        (*combinations).num_comb = num_comb;

        // array of gmer indices associated with group_srt and features_srt
        unsigned int *sortIdx = (unsigned int *) malloc(nfeat * sizeof(unsigned int));
        // sorted gmers
        unsigned int *features_srt = (unsigned int *) malloc(nfeat * g * sizeof(unsigned int));
        // gmer ids; associated with features_srt and sortIdx
        unsigned int *group_srt = (unsigned int *) malloc(nfeat * sizeof(unsigned int));
        unsigned int *cnt_comb = (unsigned int *) malloc(2 * sizeof(unsigned int)); //
        // sorted features once mismatch positions are removed
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

        // compute partial mismatch profile for these mismatch positions (slow)
        countAndUpdateTri(Ks, features_srt, group_srt, k, nfeat, total_str);

        if (approx && !skip_variance) {
            double sd = this->get_variance(Ks, K_hat, variances, n_str_pairs, n_train_pairs, iter);

            if (iter >= 1) {
                sd = std::sqrt(sd / iter);
                if (tid == 0) {
                    this->stdevs.push_back(sd);
                }
                if (delta / sd > 1.96) {
                    printf("thread %d converged in %d iterations...\n", tid, iter);
                    working = false;
                }
            }
        }
        if (approx) {
            if (max_iters != -1 && iter >= max_iters) {
                printf("thread %d reached max iterations...\n", tid);
                working = false;
            }
        }

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

        iter++;
    }

    printf("Thread %d finished in %d iterations...\n", tid, iter - 1);

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
            double val = (approx && !skip_variance) ? K_hat[j1] : Ks[j1];
            if (val != 0) Ksfinal[j1] += val;
        }
        pthread_mutex_unlock(&mutexes[num_mutex - 1]);
    } else {
        for (int i = 0; i < n_str_pairs; i++) {
            double val = (approx && !skip_variance) ? K_hat[i] : Ks[i];
            if (val != 0) Ksfinal[i] += val;
        }
    }

    free(Ks);
    if (approx && !skip_variance) {
        free(K_hat);
        free(variances);
    }
}

double *construct_test_kernel(int n_str_train, int n_str_test, double *K) {
    double* test_K = (double*) malloc(n_str_test * n_str_train * sizeof(double));
    int total_str = n_str_train + n_str_test;
    for (int i = n_str_train; i < total_str; i++){
        for (int j = 0; j < n_str_train; j++){
            test_K[(i - n_str_train) * n_str_train + j] 
                = tri_access(K, i, j) / sqrt(tri_access(K, i, i) * tri_access(K, j, j));
        }
    }
    return test_K;
}
