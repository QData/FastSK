//readinput.h

#ifndef READINPUT_H
#define READINPUT_H

#include "GakcoSVM.h"

int *string_replace (const char *s, char *d, int seqLength);
int help2();
char *readDict (char *filename,int *na);
char *parseDict (char* dataFilename, int* na);
int ** Readinput_(char *filename, char *dictfile, int *Labelout, int* len, long int *nStr, long int *maxlen, long int *minlen, int *na, GakcoSVM* object);

#endif