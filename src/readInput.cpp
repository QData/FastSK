// Code Contibution by:
//Ritambhara Singh <rs3zz@virginia.edu>
//Kamran Kowsari <kk7nc@virginia.edu >
//Arshdeep Sekhon <as5cu@virginia.edu >

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

int * string_replace (char *s, char *d);
int help2();
char * readDict (char *filename, int *na);
int dictsize;


int ** Readinput_(char *filename, char *dictFileName, int *seqLabels, int *seqLengths, long int *nStr, long int *maxlen, long int *minlen, int *dictionarySize, int maxNumStr) {
    int **output;
    char *labelfile, *seqfile;
    char *str, *linetemp, *line, *seq, *trimline, *label;
    bool isLabel = true;
    FILE *inpfile;
    char *d;
    d = readDict(dictFileName, dictionarySize);

    printf("Reading %s\n", filename);
    inpfile = fopen(filename, "r");
    
    if (inpfile) {
        line = (char *) malloc(STRMAXLEN * sizeof(char));
        int row = 0; //counts rows of the output and line number of the sequence file
        output =  (int **) malloc(maxNumStr * sizeof(int *));

        while (fgets(line, STRMAXLEN, inpfile) && row < maxNumStr) {
            linetemp = (char *) malloc(STRMAXLEN * sizeof(char *));
            strcpy(linetemp, line);
            if (isLabel) {
                label = strtok(linetemp,">");
                if(strcmp(label, "0\n") == 0) {
                    strcpy(label, "-1\n");
                }
                seqLabels[row]= atoi(label);
                isLabel = false;
            } else {
                trimline = trimwhitespace(line);
                strcpy(linetemp, trimline);
                seqLengths[row] = strlen(linetemp);
                if (seqLengths[row] > maxlen[0]) {
                    maxlen[0] = seqLengths[row];
                }
                if (seqLengths[row] < minlen[0]) {
                    minlen[0] = seqLengths[row];
                }
                output[row] = (int *) malloc(seqLengths[row] * sizeof(int));
                memset(output[row], 0, sizeof(int) * seqLengths[row]);
                strcpy(linetemp, trimline);
                seq = trimline;
                output[row] = string_replace(seq, d);
                row++;
                isLabel = true;
            }
            free(linetemp);
        }
        nStr[0] = row;
        fclose(inpfile);
        free(line);
    } else {
        perror(filename);
    }

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

int * string_replace (char *s, char *d) {
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
