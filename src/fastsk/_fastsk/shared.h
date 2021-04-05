#ifndef SHARED_H
#define SHARED_H 

#define STRMAXLEN 15000
#define MAXNSTR 15000
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <vector>

typedef struct Feature {
	int *features;
	int *group;
	int n;
	~Feature() {
		free(features);
		free(group);
	}
} Features;

typedef struct Combinations {
	int n;
	int k;
	double num_comb;
	int *comb;
} Combinations;

typedef struct Dict {
	int index;
	char *word;
} Dict;

typedef struct WorkItem {
	int m;
	int combo_num;
} WorkItem;

Features* extractFeatures(int **S, std::vector<int> seqLengths, int nStr, int g);
Features* extractFeatures(int **S, int* seqLengths, int nStr, int g);
double& tri_access(double* array, int i, int j);
unsigned int& tri_access(unsigned int* array, int i, int j, int N);
unsigned int& tri_access(unsigned int* array, int i, int j);
char *trimwhitespace(char *s);
std::string trim(std::string& s);
void cntsrtna(unsigned int *out,unsigned int *sx, int k, int r, int na);
void countAndUpdate(unsigned int *outK, unsigned int *sx, unsigned int *g, int k, int r, int nStr);
void countAndUpdateTri(unsigned int *outK, unsigned int *sx, unsigned int *g, int k, int r, int nStr);
unsigned nchoosek(unsigned n, unsigned k);
void getCombinations(unsigned int n, unsigned int k, int *pos, unsigned int depth, unsigned int margin, unsigned int *cnt_comb, unsigned int *out, int num_comb);
void shuffle(WorkItem *array, size_t n);
void print_null(const char *s);
void validate_args(int g, int m);
void g_greater_than_shortest_err(int g, int len, std::string filename);
void g_greater_than_shortest_train(int g, int len);
void g_greater_than_shortest_test(int g, int len);
double calculate_auc(double* pos, double* neg, int npos, int nneg);

#endif
