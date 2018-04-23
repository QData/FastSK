// Code Contibution by:
//Ritambhara Singh <rs3zz@virginia.edu>
//Kamran Kowsari <kk7nc@virginia.edu >
//Arshdeep Sekhon <as5cu@virginia.edu >

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <algorithm>

using namespace std;

int * string_replace (const char *s, char *d);
int help2();
char * readDict (char *filename, int *na);
int dictsize;

void labelErrorAndExit(string label) {
    printf("Error:\n");
    printf("\tInvalid label: %s\n", label.c_str());
    printf("\tPositive labels must be 1. Negative labels must be either 0 or -1\n");
    printf("\tExample label line format:\n\t >0\n");
    exit(1);
}

//Remove all white space from line
static void inline trimLine(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(),
            std::not1(std::ptr_fun<int, int>(std::isspace))));
    std::string::iterator end_pos = std::remove(s.begin(), s.end(), ' ');
    s.erase(std::find_if(s.rbegin(), s.rend(),
            std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
    s.erase(end_pos, s.end());
}

int ** Readinput_(char *filename, char *dictFileName, int *seqLabels, int *seqLengths, long int *nStr, long int *maxlen, long int *minlen, int *dictionarySize, int maxNumStr) {
    int **output;
    const char *seq;
    char *d;
    int realls = 0; //keep track of how many times we need to reallocate memory
    d = readDict(dictFileName, dictionarySize);
    
    ifstream file;
    file.open(filename);
    printf("Reading %s\n", filename);

    output = (int **) malloc(MAXNSTR * sizeof(int *));
    string line, label;
    int row = 0;
    bool isLabel = true;

    while (getline(file, line)) {
        trimLine(line);
        if (!line.empty()) {
            if (isLabel) {
                std::string::size_type pos = line.find_first_of('>');
                label = line.substr(pos + 1);
                if (label.length() > 2) labelErrorAndExit(label);
                int asNum = (stoi(label) == 0) ? -1 : stoi(label);
                if (asNum != -1 && asNum != 1) labelErrorAndExit(label);
                seqLabels[row] = asNum;
                isLabel = false;
            } else {
                int length = line.length();
                if (length > STRMAXLEN) {
                    line = line.substr(0, STRMAXLEN);
                }
                seq = line.c_str();
                seqLengths[row] = strlen(seq);
                if (seqLengths[row] > maxlen[0]) {
                    maxlen[0] = seqLengths[row];
                }
                if (seqLengths[row] < minlen[0]) {
                    minlen[0] = seqLengths[row];
                }
                output[row] = (int *) malloc(seqLengths[row] * sizeof(int));
                memset(output[row], 0, sizeof(int) * seqLengths[row]);
                output[row] = string_replace(seq, d);
                row++;
                isLabel = true;

                //allocate more space every 1000 strings after 15k
                if(row >= MAXNSTR && row % 1000 == 0){
                  realls++;
                  output = (int**)realloc(output, (MAXNSTR + 1000*realls)*sizeof(int*));
                }
            }
        }

    }
    nStr[0] = row;
    file.close();
    for (int kk = 0; kk < *nStr; kk++) {
        for(int jj = 0; jj < seqLengths[kk]; jj++) {
            if(output[kk][jj] > dictsize) {
                output[kk][jj] = 0;
            }
        }
    }
    return output;
}

// read dictionary to convert into numerical format

char * readDict (char *dictFileName, int *dictionarySize) {
    char *D;
    char *linetemp1, *line1, *next_elem, *trimline;
    int i, j;
    FILE *inpfile;

    inpfile = fopen (dictFileName, "r" );
    D = (char *) malloc(50 * sizeof(char));

    if (inpfile) {
        line1 = (char *) malloc(STRMAXLEN * sizeof(char));
        i = 0;
        while (fgets(line1, STRMAXLEN, inpfile)) {
            linetemp1 = (char *) malloc(STRMAXLEN * sizeof(char));
            trimline = trimwhitespace(line1);
            strcpy(linetemp1, trimline);
            D[i] = linetemp1[0];
            free(linetemp1);
            i++;
        }
        dictsize = i - 1;
        printf("Dictionary size = %d (+1 for uknown character)\n", dictsize + 1);
        fclose(inpfile);
        *dictionarySize = dictsize + 2;
    } else {
        perror(dictFileName);
    }
    return D;
}

//converts g-mers into numerical representation

int * string_replace (const char *s, char *d) {
    int i, count, found;
    int *array;
    found = 0;
    array = (int *) malloc(STRMAXLEN*sizeof(int));
    count = 0;
    while(s[count] != '\0') {
        for (i = 0; i <= dictsize; i++) {
            if (toupper(s[count]) == d[i]) {
                array[count]=i+1;
                found=1;
            }
        }
        if (found == 0) {
            array[count]=0;
        } else {
            found = 0;
        }
        count++;
    }
    return(array);
}

int help2() {
    printf("Usage: readInput <Input-file> <Labels-file> <Sequence-file> <Dictionary File>\n");
    return 1;
}
