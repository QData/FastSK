#include "shared.h"
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <string.h>
#include <fstream>
#include <thread>
#include <iostream>
#include <random>
#include <vector>

#define STRMAXLEN 15000
#define MAXNSTR 15000

//extract g-mers from input sequences
Features* extractFeatures(int **S, int *seqLengths, int nStr, int g) {
    int i, j, j1;
    int *group;
    int *features;
    int *s;
    int c;
    Features *F;
    int nfeat = 0;
    int sumLen = 0;
    for (i = 0; i < nStr; ++i) {
        sumLen += seqLengths[i];
        nfeat += (seqLengths[i] >= g) ? (seqLengths[i] - g + 1) : 0;
    }

    //printf("numF=%d, sumLen=%d\n", nfeat, sumLen); 
    group = (int *) malloc(nfeat * sizeof(int));
    features = (int *) malloc(nfeat * g * sizeof(int));
    c = 0;
    for (i = 0; i < nStr; ++i) {
        s = S[i];
        for (j = 0; j < seqLengths[i] - g + 1; ++j) {
            for (j1 = 0; j1 <g; ++j1) {
                features[c + j1*nfeat] = s[j + j1];
            }
            group[c] = i;
            c++;
        }
    }
    if (nfeat != c) {
        printf("Something is wrong...\n");
    }
    F = (Features *)malloc(sizeof(Features));
    (*F).features = features;
    (*F).group = group;
    (*F).n = nfeat;
    return F;
}

Features* extractFeatures(int **S, std::vector<int> seqLengths, int nStr, int g) {
    int i, j, j1;
    int *group;
    int *features;
    int *s;
    int c;
    Features *F;
    int nfeat = 0;
    int sumLen = 0;
    for (i = 0; i < nStr; ++i) {
        sumLen += seqLengths[i];
        nfeat += (seqLengths[i] >= g) ? (seqLengths[i] - g + 1) : 0;
    }

    //printf("numF=%d, sumLen=%d\n", nfeat, sumLen); 
    group = (int *) malloc(nfeat * sizeof(int));
    features = (int *) malloc(nfeat * g * sizeof(int));
    c = 0;
    for (i = 0; i < nStr; ++i) {
        s = S[i];
        for (j = 0; j < seqLengths[i] - g + 1; ++j) {
            for (j1 = 0; j1 <g; ++j1) {
                features[c + j1*nfeat] = s[j + j1];
            }
            group[c] = i;
            c++;
        }
    }
    if (nfeat != c) {
        printf("Something is wrong...\n");
    }
    F = (Features *)malloc(sizeof(Features));
    (*F).features = features;
    (*F).group = group;
    (*F).n = nfeat;
    return F;
}

// array: pointer to space (N*(N-1)/2)
// i    : row
// j    : col
// N    : length of one side
double& tri_access(double* array, int i, int j) {
    if (j > i) {
        std::swap(i, j);
    }
    return array[i * (i + 1) / 2 + j];
}

unsigned int& tri_access(unsigned int* array, int i, int j, int N) {
    if (j > i) {
        std::swap(i, j);
    }
    return array[i * (i + 1) / 2 + j];
    //return array[i*N + j];
}
unsigned int& tri_access(unsigned int* array, int i, int j) {
    if (j > i) {
        std::swap(i, j);
    }
    return array[i * (i + 1) / 2 + j];
    //return array[i*N + j];
}

char *trimwhitespace(char *str) {
    char *end;

    /* Trim leading space*/
    while (isspace(*str)) {
        str++;
    }

    if (*str == 0) {
        return str;
    }

    /* Trim trailing space*/
    end = str + strlen(str) - 1;

    while (end > str && isspace(*end)) {
        end--;
    }
    
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
void cntsrtna(unsigned int *out,unsigned int *sx, int k, int r, int na) {

    int *sxc = (int *)malloc(na*sizeof(int));
    int *bc1 = (int *)malloc(na*sizeof(int));
    int *sxl = (int *)malloc(r*sizeof( int));
    int *cc = (int *)malloc(r*sizeof(int));
    
    for (int i = 0; i < r; ++i) {
        out[i] = i;
    }
    for (int j = k - 1; j >= 0; --j) {
        for (int i = 0; i < na; ++i) {
            sxc[i] = 0;
        }
        for (int i = 0; i < r; ++i) {
            cc[i] = sx[out[i] + j*r];
            sxc[cc[i]]++;
        }

        bc1[0] = 0;
        for (int i = 1; i < na; ++i) {
            bc1[i] = bc1[i - 1] + sxc[i - 1];
        }
        for (int i = 0; i < r; ++i) {
            sxl[bc1[cc[i]]++] = out[i];
        }
        for (int i=0; i < r;++i) {
            out[i] = sxl[i];
        }
    }

    free(sxl);
    free(sxc);
    free(cc);
    free(bc1);
}

//update cumulative mismatch profile
void countAndUpdate(unsigned int *outK, unsigned int *sx, unsigned int *g, int k, int r, int nStr) {
    bool same;
    long int i, j;
    long int cu;
    long int startInd, endInd, j1;
    unsigned int *curfeat = (unsigned int *)malloc(k*sizeof(unsigned int));
    int *ucnts= (int *)malloc(nStr*sizeof(int));

    int *updind = (int *)malloc(nStr*sizeof(int));
    memset(updind, 0, sizeof(int) * nStr);
    memset(outK, 0, sizeof(unsigned int) * nStr * nStr);

    i = 0;
    while (i<r) {
        for (j = 0; j < k; ++j) {
            curfeat[j]=sx[i+j*r]; 
        }
        same = 1;
        for (j = 0;j < k; ++j) {
            if (curfeat[j]!=sx[i+j*r]) {
                same=false;
                break;
            }
        }

        same = true;
        startInd = i;
        while (same && i<r) {
            i++;
            if (i >= r) {
                break;
            }
            same = true;
            for (j = 0; j < k; ++j) {
                if (curfeat[j]!=sx[i+j*r])
                {
                    same=false;
                    break;
                }
            }
        }
        endInd = (i < r) ? (i - 1) : (r - 1);

        if ((long int) endInd - startInd + 1 > 2) {
            memset(ucnts, 0, nStr * sizeof(int));
            for (j = startInd; j <= endInd; ++j) {
                ucnts[g[j]]++;
            }
            cu = 0;
            for (j=0;j<nStr;j++) {
                if (ucnts[j]>0) {
                    updind[cu] = j;
                    cu++;
                }
            }
            for (j=0;j<cu;j++) {
                for (j1=j;j1<cu;j1++) {
                    outK[updind[j] + updind[j1] * nStr] += ucnts[updind[j]] * ucnts[updind[j1]];
                }
            }
        } else {
            for (j = startInd;j <= endInd; ++j) {
                for (j1 = startInd;j1 <= endInd; ++j1) {
                    outK[ g[j]+nStr*g[j1] ]++;
                }
            }
        }
   }
   free(updind);
   free(ucnts);
   free(curfeat);
}

//update cumulative mismatch profile for a triangular outK
void countAndUpdateTri(unsigned int *outK, unsigned int *sx, unsigned int *g, int k, int r, int nStr) {
    bool same;
    long int i, j;
    long int cu;
    long int startInd, endInd, j1;
    unsigned int *curfeat = (unsigned int *)malloc(k*sizeof(unsigned int));
    int *ucnts= (int *)malloc(nStr*sizeof(int));
    int num_str_pairs = nStr * (nStr+1) / 2;

    int *updind = (int *)malloc(nStr*sizeof(int));
    memset(updind, 0, sizeof(int) * nStr);
   
    i = 0;
    while (i<r) {
        for (j = 0; j < k; ++j)
            curfeat[j]=sx[i+j*r]; 
        same=1;
        for (j = 0;j < k; ++j)
        if (curfeat[j]!=sx[i+j*r]) {
            same=false;
            break;
        }
        same=true;
        startInd=i;
        while (same && i<r) {
            i++;
            if (i >= r) break;
            same = true;
            for (j = 0; j < k; ++j)
                if (curfeat[j]!=sx[i+j*r]) {
                    same=false;
                    break;
                }
        }
        endInd= (i<r) ? (i - 1) : (r - 1);

        if ((long int) endInd - startInd + 1 > 1) {
            memset(ucnts, 0, nStr * sizeof(int));
            for (j = startInd;j <= endInd; ++j) {
                ucnts[g[j]]++;
            }
            cu = 0;
            for (j=0;j<nStr;j++) {
                if (ucnts[j] > 0) {
                    updind[cu] = j;
                    cu++;
                }
            }
            for (j=0;j<cu;j++) {
                for (j1=j;j1<cu;j1++) {
                    tri_access(outK, updind[j1], updind[j]) += ucnts[updind[j]]*ucnts[updind[j1]];
                }
            }
        } else {
            for (j = startInd;j <= endInd; ++j) {
                for (j1 = startInd;j1 <= j; ++j1) {
                    tri_access(outK, g[j1], g[j])++;
                }
            }
        }
    }
    free(updind);
    free(ucnts);
    free(curfeat);

}

unsigned nchoosek(unsigned n, unsigned k) {
    if (k > n) return 0;
    if (k * 2 > n) k = n-k;
    if (k == 0) return 1;
    int result = n;
    for(int i = 2; i <= k; ++i ) {
        result *= (n-i+1);
        result /= i;
    }
    return result;
}

void getCombinations(unsigned int n, unsigned int k, int *pos, unsigned int depth, unsigned int margin, unsigned int* cnt_comb, unsigned int *out, int num_comb) {
    if (depth >= k) {
        for ( int j = 0; j < k; ++j) {
            out[cnt_comb[0] + j*num_comb] = pos[j];
        }
        cnt_comb[0]++;
        return;
    }

    for (int j = margin; j < n; ++j) {
        pos[depth] = j;
        getCombinations(n, k, pos, depth + 1, j + 1, cnt_comb, out, num_comb);
    }
}

//Shuffles array. used to shuffle work allocations for threads so they collide less when accumulating values into C_m
void shuffle(WorkItem *array, size_t n) {
    if (n > 1)  {
        size_t i;
        for (i = 0; i < n - 1; i++)  {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            WorkItem t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}


//null function to allow setting LIBSVM's printing to nothing on quiet mode
void print_null(const char *s) {
}

void validate_args(int g, int m) {
    if (g <= m) {
        printf("g must be greater than m\n");
        printf("Provided:\n\tg = %d\n\tm = %d\n", g, m);
        exit(1);
    }
    if (g > 20) {
        printf("g must be at most 20\n");
        printf("Provided:\n\tg = %d\n", g);
        exit(1);
    }
}

void g_greater_than_shortest_err(int g, int len, std::string filename) {
    printf("Error:\n");
    printf("\tg cannot be longer than the shortest sequence in a dataset.\n");
    printf("\tg = %d, but shortest sequence length in file %s is %d\n", g, filename.c_str(), len);
    exit(1);
}

void g_greater_than_shortest_train(int g, int len) {
    printf("Error:\n");
    printf("\tg cannot be longer than the shortest sequence in a dataset.\n");
    printf("\tg = %d, but shortest train sequence has length %d\n", g, len);
    exit(1);
}

void g_greater_than_shortest_test(int g, int len) {
    printf("Error:\n");
    printf("\tg cannot be longer than the shortest sequence in a dataset.\n");
    printf("\tg = %d, but shortest test sequence has length %d\n", g, len);
    exit(1);
}

double calculate_auc(double* pos, double* neg, int npos, int nneg) {
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
