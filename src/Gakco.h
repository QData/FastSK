
#ifndef GAKCO_H
#define GAKCO_H

void main_loop_kernel(int * elems,Features * features ,unsigned int *Ksfinal,int * cnt_k,   int *feat,int g, int k, int na,int nfeat,int nStr,int i);
void main_test_kernel(int * elems,Features * features ,unsigned int *Ksfinal,int * cnt_k,   int *feat,int g, int k, int na,int nfeat,int nStr,int nTestStr,int i);
void build_cumulative_mismatch_profiles(WorkItem *workQueue, int queueSize, int threadNum, int numThreads, int *elems, 
										Features *features, unsigned int *Ksfinal, int *cnt_k, int *feat, int g, int dictionarySize,
										int nfeat, int nStr, pthread_mutex_t *mutexes);
Features *extractFeatures(int **S, int *len, int nStr, int g);
Features* merge_features(Features* train, Features* test, int g);
double calculate_auc(double* pos, double* neg, int npos, int nneg);
int help();
int errorID1();

#endif