#ifndef UTILS_H
#define UTILS_H 

#include "shared.h"
#include <vector>
#include <string>
#include <map>

using namespace std;

class DataReader {
public:
    map<char, int> dictmap;
    vector<vector<int> > train_seq;
    vector<vector<int> > test_seq;
    vector<int> train_labels;
    vector<int> test_labels;
    int total_num_str = 0;
    int num_train_str = 0;
    int num_test_str = 0;
    long int maxlen = 0;
    long int minlen = STRMAXLEN;

    DataReader(const string, const string);
    void read_data(const string, bool);
};

static void inline trim_line(string &);
map<char, int> infer_dict(const string);
map<char, int> read_dict(const string);

#endif
