#include "svm.hpp"
#include "fastsk.hpp"
#include "shared.h"
#include "libsvm-code/libsvm.h"
#include "libsvm-code/eval.h"

#include <string>
#include <vector>
#include <iostream>

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

using namespace std;

SVM::SVM(int g, int m, double C, double nu, double eps, const string kernel_type, 
    bool quiet, double* K, int n_str_train, int n_str_test, int* test_labels, int nfeat) {

    this->g = g;
    this->m = m;
    this->k = g - m;
    this->C = C;
    this->nu = nu;
    this->eps = eps;
    validate_args(g, m);
    if (!kernel_type.empty()) {
        if (kernel_type == "linear") {
            this->kernel_type = LINEAR;
            this->kernel_type_name = "linear";
        } else if (kernel_type == "fastsk") {
            this->kernel_type = FASTSK;
            this->kernel_type_name = "fastsk";
        } else if (kernel_type == "rbf"){
            this->kernel_type = RBF;
            this->kernel_type_name = "rbf";
        } else {
            printf("Error: kernel must be: 'linear', 'fastsk', or 'rbf'\n");
            exit(1);
        }
    }
    this->n_str_train = n_str_train;
    this->n_str_test = n_str_test;
    this->total_str = n_str_train + n_str_test;
    this->test_labels = test_labels;
    this->quiet = quiet;
    this->K = K;
    this->nfeat = nfeat;
}

void SVM::fit() {

    // if ((this->kernel_type == LINEAR || this->kernel_type == RBF) && test_file.empty()) {
    //     printf("A test file must be provided for kernel type '%s'\n", this->kernel_type_name.c_str());
    //     exit(1);
    // }

    struct svm_parameter* svm_param = Malloc(svm_parameter, 1);
    svm_param->svm_type = this->svm_type;
    svm_param->kernel_type = this->kernel_type;
    svm_param->nu = this->nu;
    svm_param->gamma = 1 /(double) this->nfeat;
    svm_param->cache_size = this->cache_size;
    svm_param->C = this->C;
    svm_param->nr_weight = this->nr_weight;
    svm_param->weight_label = this->weight_label;
    svm_param->weight = this->weight;
    svm_param->shrinking = this->h;
    svm_param->probability = this->probability;
    svm_param->eps = this->eps;
    svm_param->degree = 0;

    svm_problem *prob = this->create_svm_problem(this->K, this->test_labels, svm_param);

    struct svm_model *model;

    int folds = 5;
    double cv_auc = binary_class_cross_validation(prob, svm_param, folds);
    cout << "cv_auc = " << cv_auc << endl;
    //return cv_auc;

    //model = this->train_model(this->K, this->test_labels, svm_param);
    //this->model = model;
}

svm_problem* SVM::create_svm_problem(double *K, int *labels, svm_parameter *svm_param) {
    int n_str = this->total_str;
    int n_str_train = this->n_str_train;

    struct svm_problem* prob = Malloc(svm_problem, 1);
    const char* error_msg;

    svm_node** x;
    svm_node* x_space;
    prob->l = n_str_train;
    prob->y = Malloc(double, prob->l);
    x = Malloc(svm_node*, prob->l);
    if (svm_param->kernel_type == FASTSK) {
        x_space = Malloc(struct svm_node, (n_str_train + 1) * n_str_train);
        int totalind = 0;
        for (int i = 0; i < n_str_train; i++) {
            x[i] = &x_space[totalind];
            for (int j = 0; j < n_str_train; j++) {
                x_space[j + i * (n_str_train + 1)].index = j + 1; 
                x_space[j + i * (n_str_train + 1)].value = tri_access(K, i, j);
            }
            totalind += n_str_train;
            x_space[totalind].index = -1;
            totalind++;
            prob->y[i] = labels[i];
        }
        //this->x_space = x_space;
    } else if (svm_param->kernel_type == LINEAR || svm_param->kernel_type == RBF) {
        x_space = Malloc(struct svm_node, (n_str_train + 1) * n_str_train);
        int totalind = 0;
        for (int i = 0; i < n_str_train; i++) {
            x[i] = &x_space[totalind];
            for (int j = 0; j < n_str_train; j++) {
                x_space[j + i * (n_str_train + 1)].index = j + 1; 
                x_space[j + i * (n_str_train + 1)].value = tri_access(K, i, j);
            }
            totalind += n_str_train;
            x_space[totalind].index = -1;
            totalind++;
            prob->y[i] = labels[i];
        }
        //this->x_space = x_space;
    }

    prob->x = x;

    if (this->quiet) {
        svm_set_print_string_function(&print_null);
    }

    error_msg = svm_check_parameter(prob, svm_param);

    if (error_msg) {
        fprintf(stderr, "ERROR: %s\n", error_msg);
        exit(1);
    }

    return prob;
}

svm_model* SVM::train_model(double *K, int *labels, svm_parameter *svm_param) {
    int n_str = this->total_str;
    int n_str_train = this->n_str_train;
    struct svm_problem* prob = Malloc(svm_problem, 1);
    const char* error_msg;

    svm_node** x;
    svm_node* x_space;
    prob->l = n_str_train;
    prob->y = Malloc(double, prob->l);
    x = Malloc(svm_node*, prob->l);
    if (svm_param->kernel_type == FASTSK) {
        x_space = Malloc(struct svm_node, (n_str_train + 1) * n_str_train);
        int totalind = 0;
        for (int i = 0; i < n_str_train; i++) {
            x[i] = &x_space[totalind];
            for (int j = 0; j < n_str_train; j++) {
                x_space[j + i * (n_str_train + 1)].index = j + 1; 
                x_space[j + i * (n_str_train + 1)].value = tri_access(K, i, j);
            }
            totalind += n_str_train;
            x_space[totalind].index = -1;
            totalind++;
            prob->y[i] = labels[i];
        }
        //this->x_space = x_space;
    } else if (svm_param->kernel_type == LINEAR || svm_param->kernel_type == RBF) {
        x_space = Malloc(struct svm_node, (n_str_train + 1) * n_str_train);
        int totalind = 0;
        for (int i = 0; i < n_str_train; i++) {
            x[i] = &x_space[totalind];
            for (int j = 0; j < n_str_train; j++) {
                x_space[j + i * (n_str_train + 1)].index = j + 1; 
                x_space[j + i * (n_str_train + 1)].value = tri_access(K, i, j);
            }
            totalind += n_str_train;
            x_space[totalind].index = -1;
            totalind++;
            prob->y[i] = labels[i];
        }
        //this->x_space = x_space;
    }

    prob->x = x;

    // if quiet mode, set libsvm's print function to null
    if (this->quiet) {
        svm_set_print_string_function(&print_null);
    }

    error_msg = svm_check_parameter(prob, svm_param);

    if (error_msg) {
        fprintf(stderr, "ERROR: %s\n", error_msg);
        exit(1);
    }

    // train that ish
    struct svm_model* model;
    model = svm_train(prob, svm_param);

    free(svm_param);

    return model;
}
