// Code Contibution by:
//Ritambhara Singh <rs3zz@virginia.edu>
//Kamran Kowsari <kk7nc@virginia.edu >
//Arshdeep Sekhon <as5cu@virginia.edu >

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int *string_replace (char *s, char *d);

int help2();
char *readDict (char *filename,int *na);

int dictsize;

//read input from fasta file

int ** Readinput_(char *filename,char *dictfile,int *Labelout, int* len, long int *nStr, long int *maxlen, long int *minlen,int * na)
{
  int ** output;
  char *labelfile, *seqfile;
  char *str, *linetemp, *line,*seq,*trimline,*label;
  int i,j;
  FILE *inpfile;
  char *d;
 int c =0;
  d = readDict(dictfile,na);


  printf("Reading %s\n",filename);
  inpfile = fopen ( filename, "r" );

  i=0;
  if ( inpfile )
  {
     line = (char *)malloc(STRMAXLEN*sizeof(char));
     int row=0;
     output =  (int **)malloc(STRMAXLEN * sizeof(int *));
     bool read = true;
     while(read)
     {

       if(!fgets( line, STRMAXLEN, inpfile ))
       {

	     read = false;

       }

        linetemp = (char *)malloc(STRMAXLEN*sizeof(char *));


        strcpy(linetemp,line);
	

        if(i==0)
	       {


	    
  	    label=strtok(linetemp,">");
  	    if(strcmp(label,"0\n")==0)
  	      strcpy(label,"-1\n");
  	   Labelout[row]= atoi(label);
  	    i++;
  	  }
	else
	  {  

	    trimline = trimwhitespace(line);
	  strcpy(linetemp, trimline);
	    len[row] =strlen(linetemp) ;
	   		if (len[row]>maxlen[0])
			{
			    maxlen[0] = len[row];
			}
			if (len[row]<minlen[0])
			{
			    minlen[0] = len[row];
			}

		output[row] = (int *)malloc(len[row] * sizeof(int));
    //printf("leni %d",len[row]);
    fflush(stdout);
		memset(output[row], 0, sizeof(int) * len[row]);
    strcpy(linetemp, trimline);
		seq=trimline;


		output[row]=string_replace(seq,d);
	    row++;
	    i=0;
	  }


	 free(linetemp);
     }
     nStr[0]=row;
     fclose(inpfile);
     free(line);

  }
  else
  {
    perror( filename );
  }
for (int kk =0;kk<*nStr;kk++)
{
  for(int jj=0;jj<len[kk];jj++)
  {
    if(output[kk][jj]> dictsize) output[kk][jj]=0;
  }
}
 return output;
}

// read dictionary to convert into numerical format

char *readDict (char *filename,int *na)
{
  char *D;
  char *linetemp1, *line1, *next_elem, *trimline;
  int i, j;
  FILE *inpfile;

  inpfile = fopen ( filename, "r" );
  D = (char *)malloc(50*sizeof(char));

  if ( inpfile )
  {
     line1 = (char *)malloc(STRMAXLEN*sizeof(char));
     i = 0;
     while( fgets( line1, STRMAXLEN, inpfile ) )
     {
        linetemp1 = (char *)malloc(STRMAXLEN*sizeof(char));
	trimline=trimwhitespace(line1);
        strcpy(linetemp1, trimline);

        D[i]=linetemp1[0];

	
        free(linetemp1);
        i++;;
     }
     dictsize=i-1;
     printf("Dictionary size=%d (+1 for uknown charachter)\n",dictsize+1);
     fclose(inpfile);
     *na = dictsize+2;
     
  }
  else
  {
    perror( filename );
  }

  return D;
}



//converts g-mers into numerical representation

int* string_replace (char *s, char *d)
{
  int i,count,found;
  int *array;
  found=0;
  array = (int *)malloc(STRMAXLEN*sizeof(int));
  count=0;
  while(s[count] != '\0')
    {
      for (i=0;i<=dictsize;i++)
	{
	  if(toupper(s[count])==d[i])
	    {
	      array[count]=i+1;
	      found=1;
	    }
	}
      if(found==0)
	array[count]=0;
      else
	found=0;
      count++;
    }
  return(array);
}


int help2()
{
  printf("Usage: readInput <Input-file> <Labels-file> <Sequence-file> <Dictionary File>\n");
  return 1;
}
