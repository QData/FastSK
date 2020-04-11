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
    printf("\t k : (optional) Specify a kernel filename to print to. If -l is also set, this will instead be used as the filename to load the kernel from\n");
    printf("\t o : (optional) Specify a model filename to print to. If -s is also set, this will instead be used as the filename to load the model from\n");
    printf("\t r : (optional) Kernel type. Must be linear (default), fastsk, or rbf\n");
    printf("NO ARGUMENT FLAGS\n");
    printf("\t p : (optional) Flag for model to generate probability of class. Without it, AUC can't be calculated.\n");
    printf("\t h : (optional) set to 1 or 2. If 1, will halt the program after constructing and printing out the kernel. If 2, will halt after training and printing out the model\n");
    printf("ORDERED PARAMETERS\n");
    printf("\t trainingFile : set of training examples in FASTA format\n");
    printf("\t testingFile : set of testing examples in FASTA format\n");
    printf("\t dictionaryFile : (optional) file containing the alphabet of characters that appear in the sequences. If not provided, a dictionary will be inferred from the training file.\n");
    printf("\t labelsFile : file to place labels from the test examples (simple text file)\n");
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
    double delta = 0.025;
    int max_iters = 100;
    bool skip_variance = false;
    string kernel_type = "linear";

    // SVM params
    double C = 1.0;
    double nu = 1;
    double eps = 1;

    int c;
    while ((c = getopt(argc, argv, "g:m:t:C:r:aq")) != -1) {
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
            case 'a':
                approx = true;
                break;
            case 'C':
                C = atof(optarg);
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

    FastSK* fastsk = new FastSK(g, m, t, approx, delta, max_iters, skip_variance);  
    fastsk->compute_kernel(train_file, test_file, dictionary_file);

    fastsk->fit(C, nu, eps, kernel_type);
    
    double auc = fastsk->score("auc");
}
