#pragma once

#define STRMAXLEN 15000
#include <string>
typedef struct Feature
{
	int *features;
	int *group;
	int n;
	~Feature(){
		free(features);
		free(group);
	}
} Features;

typedef struct Combinations
{
	int n;
	int k;
	double num_comb;
	int *comb;
} Combinations;

typedef struct Dict
{
	int index;
	char *word;
} Dict;

typedef struct WorkItem
{
	int m;
	int combo_num;
} WorkItem;

Features *extractFeatures(int **S, int *len, int nStr, int g);
double& tri_access(double* array, int i, int j, int N);
unsigned int& tri_access(unsigned int* array, int i, int j, int N);
char *trimwhitespace(char *s);
std::string trim(std::string& s);
void cntsrtna(unsigned int *out,unsigned int *sx, int k, int r, int na);
void countAndUpdate(unsigned int *outK, unsigned int *sx, unsigned int *g, int k, int r, int nStr);
void countAndUpdateTest(unsigned int *outK, unsigned int *sx, unsigned int *g, int k, int r, int nStr, int nTestStr);
double nchoosek(double n, double k);
void getCombinations(const int *elems, unsigned int n, unsigned int k, int *pos, unsigned int depth, unsigned int margin, unsigned int *cnt_comb, unsigned int *out, int num_comb);
