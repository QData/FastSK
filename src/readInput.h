//readinput.h

#ifndef READINPUT_H
#define READINPUT_H

int *string_replace (char *s, char *d);

int help2();
char *readDict (char *filename,int *na);
//int dictsize;
int ** Readinput_(char *filename,char *dictfile,int *Labelout, int* len, long int *nStr, long int *maxlen, long int *minlen,int * na);

#endif