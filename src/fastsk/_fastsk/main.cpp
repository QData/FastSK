#include "fastsk.hpp"
#include <string>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

using namespace std;

int help() {
    printf("\nUsage: fastsk [options] <trainingFile> <testFile> <dictionaryFile> <labelsFile>\n");
    printf("FLAGS WITH ARGUMENTS\n");
    printf("\t g : gmer length; length of substrings (allowing up to m mismatches) used to compare sequences. Constraints: 0 < g < 20\n");
    printf("\t m : maximum number of mismatches when comparing two gmers. Constraints: 0 <= m < g\n");
    printf("\t t : (optional) number of threads to use. Set to 1 to not multithread kernel computation\n");
    printf("\t C : (optional) SVM C parameter. Default is 1.0\n");
    printf("\t r : (optional) Kernel type. Must be linear (default), fastsk, or rbf\n");
    printf("\t I : (optional) Maximum number of iterations. Default 100. The number of mismatch positions to sample when running the approximation algorithm.\n");
    printf("NO ARGUMENT FLAGS\n");
    printf("\t a : (optional) Approximation. If set, the fast approximation algorithm will be used to compute the kernel function\n");
    printf("\t q : (optional) Quiet mode. If set, Kernel computation and SVM training info won't be printed.\n");
    printf("ORDERED PARAMETERS\n");
    printf("\t trainingFile : set of training examples in FASTA format\n");
    printf("\t testingFile : set of testing examples in FASTA format\n");
    printf("\t dictionaryFile : (optional) file containing the alphabet of characters that appear in the sequences. If not provided, a dictionary will be inferred from the training file.\n");
    printf("\n");
    printf("\nExample usage:\n");
    printf("\t./fastsk -g 8 -m 4 -t 4 -C 0.01 1.1.train.fasta 1.1.test.fasta protein_dictionary.txt\n\n");

    return 1;
}

int main(int argc, char* argv[]) {
    int quiet = 0;
    string train_file;
    string test_file;
    string dictionary_file;

    // Kernel function params
    int g = -1;
    int m = -1;
    int t = 20;
    bool approx = false;
    int max_iters = 100;
    double delta = 0.025;
    bool skip_variance = false;
    string kernel_type = "linear";

    // SVM params
    double C = 1.0;
    double nu = 1;
    double eps = 1;

    int c;
    while ((c = getopt(argc, argv, "g:m:t:I:C:r:aq")) != -1) {
        switch (c) {
            case 'g':
                g = atoi(optarg);
                break;
            case 'm':
                m = atoi(optarg);
                break;
            case 't':
                t = atoi(optarg);
                break;
            case 'I':
                t = atoi(optarg);
                break;
            case 'C':
                C = atof(optarg);
                break;
            case 'a':
                approx = true;
                break;
            case 'r':
                kernel_type = optarg;
                break;
            case 'q':
                quiet = 1;
                break;
            break;
        }
    }

    if (g == -1) {
        printf("Must provide a value for the g parameter\n");
        return help();
    }
    if (m == -1) {
        printf("Must provide a value for the m parameter\n");
        return help();
    }
    if (kernel_type != "linear" && kernel_type != "fastsk" && kernel_type != "rbf") {
        printf("kernel must be linear, fastsk, or rbf.\n");
        return help();
    }

    int arg_num = optind;

    if (arg_num < argc) {
        train_file = argv[arg_num++];
    } else {
        printf("Train data file required\n");
        return help();
    }
    if (arg_num < argc) {
        test_file = argv[arg_num++];
    } else {
        printf("Test data file required\n");
        return help();
    }
    if (arg_num < argc) {
        dictionary_file = argv[arg_num++];
    }

    //FastSK* fastsk = new FastSK(g, m, t, approx, delta, max_iters, skip_variance);  
    //fastsk->compute_kernel(train_file, test_file, dictionary_file);

    //fastsk->fit(C, nu, eps, kernel_type);
    
    //double auc = fastsk->score("auc");
}
