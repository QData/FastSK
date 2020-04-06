#include "dataset.hpp"
#include "shared.h"
#include <map>
#include <iostream>
#include <fstream>
#include <vector>
#include <assert.h>
#include <string.h>
#include <algorithm>

//Remove all white space from line
static void inline trimLine(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(),
            std::not1(std::ptr_fun<int, int>(std::isspace))));
    std::string::iterator end_pos = std::remove(s.begin(), s.end(), ' ');
    s.erase(std::find_if(s.rbegin(), s.rend(),
            std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
    s.erase(end_pos, s.end());
}

void labelErrorAndExit(std::string label) {
    printf("Error:\n");
    printf("\tInvalid label: %s\n", label.c_str());
    printf("\tPositive labels must be 1. Negative labels must be either 0 or -1\n");
    printf("\tExample label line format:\n\t >0\n");
    exit(1);
}

//converts sequences into numerical representation
int * string_replace (const char *s, char *d, int seqLength) {
    int i, count, found;
    int *array;
    int dictsize = strlen(d);
    found = 0;
    array = (int *) malloc(seqLength * sizeof(int));
    memset(array, 0, seqLength * sizeof(int));
    count = 0;
    while(s[count] != '\0') {
        for (i = 0; i <= dictsize; i++) {
            if (toupper(s[count]) == toupper(d[i])) {
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

Dataset::Dataset(std::string filename, bool is_test_set, char *dict, std::string dictFileName) { 
    this->filename = filename;
    this->is_test_set = is_test_set;

    /*
    If this is a test dataset, set the dictionary by providing a char *dict to use.
    If it's a train dataset, we can either:
        (1) read a provided dictionary file
        (2) build a dictionary by reading the training file (filename)
    */

    if (is_test_set) {
        if (dict == NULL) {
            printf("char *dict must be provided for a test dataset!\n");
            exit(1);
        }
        char *copy = (char *) malloc((strlen(dict) + 1) * sizeof(char));
        strcpy(copy, dict);
        this->dict = copy;
        this->dictionarySize = strlen(this->dict) + 1;
    } else {
        if (this->dictFileName.empty()) {
            // if no dictionary provided, construct one from the input file
            this->parseDict();
            printf("Dictionary built from %s\n", this->filename.c_str());
        } else {
            // if a dictionary is provided, read from it
            this->readDict();
            printf("Dictionary built from %s\n", this->dictFileName.c_str());
        }
    }

    printf("Dictionary: %s\n", this->dict);
    printf("Size: %d (+1 for unknown character)\n", this->dictionarySize);
}

void Dataset::collect_data(bool quiet) {
    std::string filename = this->filename;
    // matrix of numerical sequences
    const char *seq;
    // the dictionary
    //keep track of how many times we need to reallocate memory
    int realls = 0;

    printf("Collecting data from: %s\n", filename.c_str());

    std::ifstream file;
    file.open(filename.c_str());
    if (file.fail()) {
        perror(filename.c_str());
        exit(1);
    }

    this->S = (int **) malloc(MAXNSTR * sizeof(int *));
    std::string line, label;
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
                this->seqLabels[row] = asNum;
                isLabel = false;
            } else {
                int length = line.length();
                if (length > STRMAXLEN) {
                    line = line.substr(0, STRMAXLEN);
                }
                seq = line.c_str();
                this->seqLengths[row] = strlen(seq);
                if (this->seqLengths[row] > this->maxlen) {
                    this->maxlen = this->seqLengths[row];
                }
                if (this->seqLengths[row] < this->minlen) {
                    this->minlen = this->seqLengths[row];
                }
                this->S[row] = string_replace(seq, this->dict, this->seqLengths[row]);
                row++;
                isLabel = true;
                //allocate more space every 1000 strings after 15k
                if(row >= MAXNSTR && row % 1000 == 0){
                    realls++;
                    this->S = (int**)realloc(this->S, (MAXNSTR + 1000*realls)*sizeof(int*));
                    this->seqLengths = (int*)realloc(this->seqLengths, (MAXNSTR + 1000*realls)*sizeof(int));
                    this->seqLabels = (int*)realloc(this->seqLabels, (MAXNSTR + 1000*realls)*sizeof(int));
                }
            }
        }
    }
    //final realloc to relieve memory waste
    this->S = (int**) realloc(this->S, row*sizeof(int*));
    this->seqLengths = (int*) realloc(this->seqLengths, row*sizeof(int));
    this->seqLabels = (int*) realloc(this->seqLabels, row*sizeof(int));

    this->nStr = row;
    file.close();
    for (int kk = 0; kk < this->nStr; kk++) {
        for(int jj = 0; jj < this->seqLengths[kk]; jj++) {
            if(this->S[kk][jj] > this->dictionarySize) {
                this->S[kk][jj] = 0;
            }
        }
    }

    if (!quiet) {
        if (maxlen != minlen) {
            printf("Read %ld strings of max length = %ld and min length=%ld\n", nStr, maxlen, minlen);
        } else {
            printf("Read %ld strings of length = %ld\n", nStr, maxlen);
        }
    }
}

void Dataset::parseDict(){
    char* D; //The dictionary
    std::ifstream file;
    file.open(this->filename);
    if (file.fail()) {
        perror(filename.c_str());
        exit(1);
    }
    std::string line;
    //hope the datafile starts with label as per proper format
    bool isLabel = true;
    //set int equal to 1 when we encounter a unique char
    std::map<char, int> dictmap;

    while (getline(file, line)) {
        trimLine(line);
        if (!line.empty()) {
            if (isLabel) {
                isLabel = false;
            } else {
                int length = line.length();
                for (int i = 0; i < length; i++){
                    if (dictmap[toupper(line[i])] > 0)
                        dictmap[toupper(line[i])]++;
                    else
                        dictmap[toupper(line[i])] = 1;
                }
                isLabel = true;
            }
        }
    }
    file.close();
    int i = 0;
    D = (char*) malloc(140 * sizeof(char));
    for (std::map<char, int>::iterator it = dictmap.begin(); it != dictmap.end(); it++) {
        D[i] = it->first;
        i++;
    }
    D[i] = '\0';
    this->dictionarySize = i + 1;
    D = (char*) realloc(D, (i + 1) * sizeof(char));
    this->dict = D;
}

void Dataset::readDict() {
    char *linetemp1, *line1, *trimline;
    int i, dictsize;
    FILE *inpfile = fopen((this->dictFileName).c_str(), "r" );
    char *D = (char *) malloc(150 * sizeof(char));
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
        fclose(inpfile);
        free(line1);
        this->dictionarySize = dictsize + 2;
        D[i] = '\0';
        D = (char*) realloc(D, (i + 1) * sizeof(char));
    } else {
        perror(dictFileName.c_str());
        exit(1);
    }
    this->dict = D;
}

/*Not great design, but this should only be called after
we know no longer need the sequences and sequence lengths anymore.
I.e., after we've extracted the features.*/
void Dataset::free_strings() {
    for(int i = 0; i < this->nStr; i++){
        if(this->S[i] != NULL)
            free(this->S[i]);
    }
    free(this->seqLengths);
    free(this->S);
}

ArrayDataset::ArrayDataset(std::vector<std::string> sequences, std::vector<int> labels) {
    this->sequences = sequences;
    this->labels = labels;
    this->n_str = sequences.size();
    assert(sequences.size() == labels.size());
}

void ArrayDataset::read_data() {
    char* D;
    std::string sequence;
    //set int equal to 1 when we encounter a unique char
    std::map<char, int> dictmap;
    for (auto iter = this->sequences.begin(); iter != this->sequences.end(); iter++) {
        std::string str = *iter;
        int length = str.length();
        if (length < this->minlen) {
            this->minlen = length;
        }
        if (length > this->maxlen) {
            this->maxlen = length;
        }
        this->seqLengths.push_back(length);
        for (int i = 0; i < length; i++){
            if (dictmap[toupper(str[i])] > 0)
                dictmap[toupper(str[i])]++;
            else
                dictmap[toupper(str[i])] = 1;
        }
    }
    int i = 0;
    D = (char*) malloc(140 * sizeof(char));
    for (std::map<char, int>::iterator it = dictmap.begin(); it != dictmap.end(); it++) {
        D[i] = it->first;
        i++;
    }
    D[i] = '\0';
    this->dictionarySize = i + 1;
    D = (char*) realloc(D, (i + 1) * sizeof(char));
    this->dict = D;
    printf("Dictionary: %s\n", this->dict);
    printf("Size: %d (+1 for unknown character)\n", this->dictionarySize);
}

void ArrayDataset::numericize_seqs() {
    const int n_str = this->n_str;
    printf("n_str = %d\n", n_str);
    this->S = (int **) malloc(n_str * sizeof(int *));
    const char *seq;
    for (int i = 0; i < n_str; i++) {
        std::string str = this->sequences[i];
        int length = this->seqLengths[i];
        seq = str.c_str();
        this->S[i] = string_replace(seq, this->dict, this->seqLengths[i]);
    }
}

TestArrayDataset::TestArrayDataset(std::vector<std::string> sequences, 
    std::vector<int> labels, char *dict) : ArrayDataset(sequences, labels) {
    char *copy = (char *) malloc((strlen(dict) + 1) * sizeof(char));
    strcpy(copy, dict);
    this->dict = copy;
    this->dictionarySize = strlen(this->dict);
}
