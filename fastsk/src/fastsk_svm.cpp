#include "fastsk_svm.hpp"
#include "fastsk_kernel.hpp"
#include "shared.h"
#include "libsvm-code/svm.h"
#include "libsvm-code/eval.h"

#include <string>
#include <vector>
#include <iostream>

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

using namespace std;

FastSK_SVM::FastSK_SVM(int g, int m, double C, double nu, double eps, const string kernel_type, 
    bool quiet, double* K, int n_str_train, int n_str_test, int* train_labels, int* test_labels,
    int nfeat) {

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
    this->train_labels = train_labels;
    this->test_labels = test_labels;
    this->quiet = quiet;
    this->K = K;
    this->nfeat = nfeat;
}

void FastSK_SVM::fit() {

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

    svm_problem *prob = this->create_svm_problem(this->K, this->train_labels, svm_param);

    struct svm_model *model;
    model = this->train_model(this->K, this->train_labels, svm_param);
    this->model = model;
}

double FastSK_SVM::cv(int folds) {

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

    svm_problem *prob = this->create_svm_problem(this->K, this->train_labels, svm_param);
    double cv_auc = binary_class_cross_validation(prob, svm_param, folds);

    return cv_auc;
}

double FastSK_SVM::score(string metric) {
    if (metric != "accuracy" && metric != "auc") {
        throw std::invalid_argument("metric argument must be 'accuracy' or 'auc'");
    }
    int n_str = this->total_str;
    int n_str_train = this->n_str_train;
    int n_str_test = this->n_str_test;
    printf("Predicting labels for %d sequences...\n", n_str_test);
    double *test_K = construct_test_kernel(n_str_train, n_str_test, this->kernel);
    int *test_labels = this->test_labels;
    printf("Test kernel constructed...\n");

    int num_sv = this->model->nSV[0] + this->model->nSV[1];
    printf("num_sv = %d\n", num_sv);
    struct svm_node *x = Malloc(struct svm_node, n_str_train + 1);
    int correct = 0;
    // aggregators for finding num of pos and neg samples for auc
    int pagg = 0, nagg = 0;
    double* neg = Malloc(double, n_str_test);
    double* pos = Malloc(double, n_str_test);

    int fp = 0, fn = 0; //counters for false postives and negatives
    int tp = 0, tn = 0; //counters for true postives and negatives
    int labelind = 0;
    for (int i =0; i < 2; i++){
        if (this->model->label[i] == 1)
            labelind = i;
    }

    FILE *auc_file;
    auc_file = fopen("auc_file.txt", "w+");

    int svcount = 0;
    for (int i = 0; i < n_str_test; i++) {
        if (this->kernel_type == FASTSK) {
            for (int j = 0; j < n_str_train; j++){
                x[j].index = j + 1;
                x[j].value = 0;
            }
            svcount = 0;
            for (int j = 0; j < n_str_train; j++){
                if (j == this->model->sv_indices[svcount] - 1){
                    x[j].value = test_K[i * num_sv + svcount];
                    svcount++;
                }
            }
            x[n_str_train].index = -1;
        } else if (this->kernel_type == LINEAR || this->kernel_type == RBF) {
            for (int j = 0; j < n_str_train; j++){
                x[j].index = j + 1;
                x[j].value = test_K[i * n_str_train + j];
            }
            x[n_str_train].index = -1;
        }

        // probs = [prob_pos, prob_neg], not [prob_neg, prob_pos]
        double probs[2];
        double guess = svm_predict_probability(this->model, x, probs);
        fprintf(auc_file, "%d,%f\n", test_labels[i], probs[0]);

        if (test_labels[i] > 0) {
            pos[pagg] = probs[labelind];
            pagg += 1;
            if (guess < 0) {
                fn++;
            } else {
                tp++;
            }
        } else {
            neg[nagg] = probs[labelind];
            nagg += 1;
            if (guess > 0) {
                fp++;
            } else {
                tn++;
            }
        }

        //printf("guess = %f and test_labels[%d] = %d\n", guess, i, test_labels[i]);
        if ((guess < 0.0 && test_labels[i] < 0) || (guess > 0.0 && test_labels[i] > 0)) {
            correct++;
        }
    }

    fclose(auc_file);

    if (pagg == 0 && metric == "auc") {
        printf("No positive examples were in the test set. AUROC is undefined in this case.\n");
    }

    double tpr = tp / (double) pagg;
    double tnr = tn / (double) nagg;
    double fnr = fn / (double) pagg;
    double fpr = fp / (double) nagg;
    double auc = calculate_auc(pos, neg, pagg, nagg);
    double acc = correct / (double)  n_str_test;
    if (!this->quiet) {
        printf("Num sequences: %d\n", nagg + pagg);
        printf("Num positive: %d, Num negative: %d\n", pagg, nagg);
        printf("TPR: %f\n", tpr);
        printf("TNR: %f\n", tnr);
        printf("FNR: %f\n", fnr);
        printf("FPR: %f\n", fpr);
    }
    printf("\nAccuracy: %f\n", acc);
    printf("AUROC: %f\n", auc);

    if (metric == "auc") {
        return auc;
    }
    
    return acc;
}

svm_problem* FastSK_SVM::create_svm_problem(double *K, int *labels, svm_parameter *svm_param) {
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

svm_model* FastSK_SVM::train_model(double *K, int *labels, svm_parameter *svm_param) {
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
