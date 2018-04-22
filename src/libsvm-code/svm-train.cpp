// Modified from original by Andrew Norton (apn4za)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <assert.h>
#include "svm.h"
#include "../gakco-svm.h"
#include "../readInput.h"
#include "../shared.h"
#include "../GaKCo.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

void print_null(const char *s) {
}

void exit_with_help(char* exe_name) {
  printf("Usage: %s [options] dictionary_file training_set_file [model_file]\n", exe_name);
  printf("kernel options:\n"
      "  -g gap_length     set gap length in kernel function (default 7); must be > 0 and < 20\n"
      "  -k kmer_length    set length of k-mer; must be < g (default 5) and g-k must be < 20\n"
      "  -p parallel       use multithreading if 1 [default 1]\n"
      "svm options:\n"
      "  -s svm_type       set type of SVM (default 0)\n"
      "                      0 -- C-SVC (multi-class classification)\n"
      "                      1 -- nu-SVC (multi-class classification)\n"
      /*"                      3 -- epsilon-SVR (regression)\n" // TODO restore SVR functionality
        "                      4 -- nu-SVR (regression)\n"*/
      "  -c cost           set the parameter C of C-SVC (default 1)\n"
      "  -n nu             set the parameter nu of nu-SVC (default 0.5)\n"
      "  -m cachesize      set cache memory size in MB (default 100)\n"
      "  -e epsilon        set tolerance of termination criterion (default 0.001)\n"
      "  -h shrinking      whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
      "  -b probability_estimates   whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"
      "  -wi weight        set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
      "  -v n              n-fold cross validation mode\n"
      "  -q                quiet mode (no outputs)\n");
  exit(1);
}

void exit_input_error(int line_num) {
  fprintf(stderr, "Wrong input format at line %d\n", line_num);
  exit(1);
}

/* Function declarations */
void parse_command_line(int argc, char **argv, char *dictionary_file_name,
    char *input_file_name, char *model_file_name);
void read_problem(const char *dictionary_filename, const char *filename, int g, int k, int parallel);
void do_cross_validation();

/* Globals */
struct gakco_parameter k_param;  // kernel parameters; set by parse_command_line
struct svm_parameter param;      // set by parse_command_line
struct svm_problem prob;         // set by read_problem
struct svm_model *model;
struct svm_node *x_space;
int cross_validation;
int nr_fold;

/* Globals from GaKCo */
long int gakco_num_strings = 0;
long int gakco_max_len     = 0;
long int gakco_min_len     = 1000;
int gakco_alpha_size       = 0;
double* gakco_kernel_matrix  = NULL;

static char *line = NULL;
static int max_line_len;

static char *readline(FILE * input) {
  int len;

  if (fgets(line, max_line_len, input) == NULL)
    return NULL;

  while (strrchr(line, '\n') == NULL) {
    max_line_len *= 2;
    line = (char *)realloc(line, max_line_len);
    len = (int)strlen(line);
    if (fgets(line + len, max_line_len - len, input) == NULL)
      break;
  }
  return line;
}


void do_cross_validation() {
  int i;
  int total_correct = 0;
  double total_error = 0;
  double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
  double *target = Malloc(double, prob.l);

  svm_cross_validation(&prob, &param, nr_fold, target);
  if (param.svm_type == EPSILON_SVR || param.svm_type == NU_SVR) {
    for (i = 0; i < prob.l; i++) {
      double y = prob.y[i];
      double v = target[i];
      total_error += (v - y) * (v - y);
      sumv += v;
      sumy += y;
      sumvv += v * v;
      sumyy += y * y;
      sumvy += v * y;
    }
    printf("Cross Validation Mean squared error = %g\n",
        total_error / prob.l);
    printf
      ("Cross Validation Squared correlation coefficient = %g\n",
       ((prob.l * sumvy - sumv * sumy) * (prob.l * sumvy -
         sumv * sumy)) /
       ((prob.l * sumvv - sumv * sumv) * (prob.l * sumyy -
         sumy * sumy))
      );
  } else {
    for (i = 0; i < prob.l; i++)
      if (target[i] == prob.y[i])
        ++total_correct;
    printf("Cross Validation Accuracy = %g%%\n",
        100.0 * total_correct / prob.l);
  }
  free(target);
}

void parse_command_line(int argc, char **argv, char *dictionary_file_name, 
    char *input_file_name, char *model_file_name) {
    
  int i;
  void (*print_func) (const char *) = NULL;  // default 
  // printing 
  // to
  // stdout

  // gakco kernel default values
  k_param.gap_length   = 7; // g
  k_param.kmer_length  = 5; // k
  k_param.use_parallel = true;

  // default values
  param.svm_type       = C_SVC;
  param.kernel_type    = GAKCO;
  param.nu             = 0.5;
  param.cache_size     = 100;
  param.C              = 1;
  param.eps            = 1e-3;
  param.shrinking      = 1;
  param.probability    = 0;
  param.nr_weight      = 0;
  param.weight_label   = NULL;
  param.weight         = NULL;
  cross_validation     = 0;

  // parse options TODO replace with arg parsing library
  for (i = 1; i < argc; i++) {
    if (argv[i][0] != '-')
      break;
    if (++i >= argc)
      exit_with_help(argv[0]);
    switch (argv[i - 1][1]) {
      case 'k':
        k_param.kmer_length = atoi(argv[i]);
        break;
      case 'g':
        k_param.gap_length = atoi(argv[i]);
        break;
      case 's':
        param.svm_type = atoi(argv[i]);
        break;
      case 'n':
        param.nu = atof(argv[i]);
        break;
      case 'm':
        param.cache_size = atof(argv[i]);
        break;
      case 'c':
        param.C = atof(argv[i]);
        break;
      case 'e':
        param.eps = atof(argv[i]);
        break;
      case 'h':
        param.shrinking = atoi(argv[i]);
        break;
      case 'b':
        param.probability = atoi(argv[i]);
        break;
      case 'q':
        print_func = &print_null;
        i--;
        break;
      case 'v':
        cross_validation = 1;
        nr_fold = atoi(argv[i]);
        if (nr_fold < 2) {
          fprintf(stderr,
              "n-fold cross validation: n must >= 2\n");
          exit_with_help(argv[0]);
        }
        break;
      case 'w':
        ++param.nr_weight;
        param.weight_label =
          (int *)realloc(param.weight_label,
              sizeof(int) * param.nr_weight);
        param.weight =
          (double *)realloc(param.weight,
              sizeof(double) * param.nr_weight);
        param.weight_label[param.nr_weight - 1] =
          atoi(&argv[i - 1][2]);
        param.weight[param.nr_weight - 1] = atof(argv[i]);
        break;
      default:
        fprintf(stderr, "Unknown option: -%c\n",
            argv[i - 1][1]);
        exit_with_help(argv[0]);
    }
  }

  svm_set_print_string_function(print_func);

  // Get the dictionary filename
  if (i >= argc)
    exit_with_help(argv[0]);
  strcpy(dictionary_file_name, argv[i]);
  i++;

  // Get training filename
  if (i >= argc)
    exit_with_help(argv[0]);
  strcpy(input_file_name, argv[i]);

  // Get output filename (optional)
  if (i < argc - 1)
    strcpy(model_file_name, argv[i + 1]);
  else {
    char *p = strrchr(argv[i], '/');
    if (p == NULL)
      p = argv[i];
    else
      ++p;
    sprintf(model_file_name, "%s.model", p);
  }
}

// read in a problem (in FASTA format)
// heavily modified from libsvm (really nothing is the same)
void read_problem(const char *dictionary_filename, const char *filename, int g, int k, int parallel) {
  int*  labels_array   = (int*)malloc(STRMAXLEN*sizeof(int));
  int*  len_array      = (int*)malloc(STRMAXLEN*sizeof(int));

  // TODO check how many need to be global.
  int** gakco_input_matrix = read_input(filename,
      dictionary_filename,
      labels_array,
      len_array,
      &gakco_num_strings,
      &gakco_max_len,
      &gakco_min_len,
      &gakco_alpha_size);

  // Size of svm problem is the number of strings in the file
  prob.l = gakco_num_strings;

  gakco_kernel_matrix = compute_kernel_matrix(gakco_input_matrix, len_array, g, k, 
      gakco_num_strings, gakco_alpha_size, parallel);

  prob.y = Malloc(double, prob.l);
  prob.x = Malloc(struct svm_node *, prob.l);
  x_space = Malloc(struct svm_node, prob.l); // Kind-of hacky, but we're just going to have 1 node per thing.

  // copy over the labels array
  // Fill the x matrix with just indices into the kernel matrix
  for (int i = 0; i < prob.l; i++) {
    prob.y[i] = labels_array[i];
    prob.x[i] = &x_space[i];
    prob.x[i]->index = i;
    prob.x[i]->value = i;
  }

  free(labels_array);
  free(len_array);
}
