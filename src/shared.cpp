//Code Contributions by:
// Code Contibution by:
//Ritambhara Singh <rs3zz@virginia.edu>
//Kamran Kowsari <kk7nc@virginia.edu >
//Arshdeep Sekhon <as5cu@virginia.edu >
//Modified version of originial implementation by Pavel Kuksa (for distance based kernel)

#include "shared.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <malloc.h>
#include <string.h>
#include <fstream>
#include <thread>
#include <iostream>
#include <random>


char *trimwhitespace(char *s);


// array: pointer to space (N*(N-1)/2)
// i    : row
// j    : col
// N    : length of one side
double& tri_access(double* array, int i, int j, int N) {
	if (j > i)
		std::swap(i, j);
	return array[i*(i+1)/2 + j];
	//return array[i*N + j];
	}

unsigned int& tri_access(unsigned int* array, int i, int j, int N) {
	if (j > i)
		std::swap(i, j);
	return array[i*(i+1)/2 + j];
	//return array[i*N + j];
}

char *trimwhitespace(char *str)
{
	char *end;

	/* Trim leading space*/
	while (isspace(*str)) str++;

	if (*str == 0)
		return str;

	/* Trim trailing space*/
	end = str + strlen(str) - 1;

	while (end > str && isspace(*end)) end--;

	
	*(end + 1) = 0;

	return str;
}

//const& so the string passed in is not affected
//trims whitespaces at either end of string.
std::string trim(std::string& str) {
    if(str.empty())
        return str;

    size_t firstScan = str.find_first_not_of(' ');
    size_t first = firstScan == std::string::npos ? str.length() : firstScan;
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, last-first+1);
}

// count and sort
void cntsrtna(unsigned int *out,unsigned int *sx, int k, int r, int na)
{

	int *sxc = (int *)malloc(na*sizeof(int));
	int *bc1 = (int *)malloc(na*sizeof(int));
	int *sxl = (int *)malloc(r*sizeof( int));
	int *cc = (int *)malloc(r*sizeof(int));
	

		for (int i = 0; i < r; ++i)
			out[i] = i;
		for (int j = k - 1; j >= 0; --j)
		{
			for (int i = 0; i < na; ++i)
				sxc[i] = 0;
			for (int i = 0; i < r; ++i)
			{
				cc[i] = sx[out[i] + j*r];
				
				sxc[cc[i]]++;
			}

			bc1[0] = 0;
			for (int i = 1; i < na; ++i)
				bc1[i] = bc1[i - 1] + sxc[i - 1];
			for (int i = 0; i < r; ++i)
			{
			

				sxl[bc1[cc[i]]++] = out[i];
			}
			for (int i=0; i < r;++i)
			{
				out[i] = sxl[i];
				}


		}

free(sxl);
free(sxc);
free(cc);
free(bc1);
}

//update cumulative mismatch profile
void countAndUpdate(unsigned int *outK, unsigned int *sx, unsigned int *g, int k, int r, int nStr)
{
   bool same;
   long int i, j;
   long int cu;
   long int startInd, endInd, j1;
   int *curfeat = (int *)malloc(k*sizeof(int));
   int *ucnts= (int *)malloc(nStr*sizeof(int));

   int *updind = (int *)malloc(nStr*sizeof(int));
   memset(updind, 0, sizeof(int) * nStr);
   memset(outK, 0, sizeof(unsigned int) * nStr * nStr);
   
   i = 0;
   while (i<r)
   {
		for (j = 0; j < k; ++j)
			curfeat[j]=sx[i+j*r]; 
     	same=1;
		for (j = 0;j < k; ++j)
		if (curfeat[j]!=sx[i+j*r])
		{
			same=false;
			break;
		}

		same=true;
		startInd=i;
		while (same && i<r)
		{
			i++;
			if (i >= r) break;
			same = true;
			for (j = 0; j < k; ++j)
				if (curfeat[j]!=sx[i+j*r])
				{
					same=false;
					break;
				}
		}
		endInd= (i<r) ? (i-1):(r-1);

		if ((long int)endInd-startInd+1>2)
		{
			memset(ucnts, 0, nStr * sizeof(int));
	        for (j = startInd;j <= endInd; ++j)  ucnts[g[j]]++;
			cu = 0;
			for (j=0;j<nStr;j++)
			{
				if (ucnts[j]>0)
				{
					updind[cu] = j;
					cu++;
				}
	      	}
			for (j=0;j<cu;j++)
			{
			    for (j1=j;j1<cu;j1++)
				{
					outK[updind[j]+updind[j1]*nStr]+=ucnts[updind[j]]*ucnts[updind[j1]];
				}
			}
		}
		else
		{
			for (j = startInd;j <= endInd; ++j)
				for (j1 = startInd;j1 <= endInd; ++j1)
					outK[ g[j]+nStr*g[j1] ]++;
		}
   }
  free(updind);
  free(ucnts);
  free(curfeat);

}

void countAndUpdateTest(unsigned int *outK, unsigned int *sx, unsigned int *g, int k, int r, int nStr, int nTestStr){
   bool same;
   long int i, j;
   long int cu;
   long int startInd, endInd, j1;
   int *curfeat = (int *)malloc(k*sizeof(int));
   int *ucnts= (int *)malloc(nStr*sizeof(int));

   int *updind = (int *)malloc(nStr*sizeof(int));
   memset(updind, 0, sizeof(int) * nStr);
   memset(outK, 0, sizeof(unsigned int) * nStr * nTestStr);
}


double nchoosek(double n, double k)
{
	int i;
	double *nums, *dens;
	double prod;
	if (k > n / 2) k = n - k;
	if (k == 0) return 1;
	if (k == 1) return n;
	if (k > 1)
	{
		nums = (double*)malloc(k*sizeof(double));
		dens = (double*)malloc(k*sizeof(double));
		for (i = 0; i < k; ++i)
		{
			nums[i] = n - k + 1 + i;
			dens[i] = i + 1;
		}
		prod = 1;
		for (i = 0; i < k; ++i)
		{
			nums[i] /=  dens[i];
			prod = prod * nums[i];
		}
		free(nums); free(dens);
	}
	return prod;
}

void getCombinations( int *elems,unsigned int n, unsigned int k, int *pos, unsigned int depth, unsigned int margin, unsigned int* cnt_comb, unsigned int *out, int num_comb)
{


		if (depth >= k)
		{
				for ( int j = 0; j < k; ++j)
				{
					out[cnt_comb[0] + j*num_comb] = pos[j];
				}
				cnt_comb[0]++;
				return;
		}


			for ( int j = margin; j < n; ++j)
			{
				pos[depth] = j;
				getCombinations(elems,n, k, pos, depth + 1, j + 1, cnt_comb, out, num_comb);
			}


}





