#include "utils.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <set>
#include <map>
#include <assert.h>
#include <stdexcept>
#include <algorithm>

using namespace std;

// remove all white space from line
static void inline trim_line(string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(),
            std::not1(std::ptr_fun<int, int>(std::isspace))));
    std::string::iterator end_pos = std::remove(s.begin(), s.end(), ' ');
    s.erase(std::find_if(s.rbegin(), s.rend(),
            std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
    s.erase(end_pos, s.end());
}

void labelErrorAndExit(std::string label) {
    printf("Error:\n");
    printf("\tInvalid label: %s\n", label.c_str());
    printf("\tPositive labels must be 1. Negative labels must be either 0 or -1\n");
    printf("\tExample label line format:\n\t >0\n");
    exit(1);
}

// create dictionary by reading from a dictionary file
map<char, int> read_dict(const string dictionary_file) {
    ifstream file;
    map<char, int> dictmap;
    string line;
    int key_val = 1;

    file.open(dictionary_file);
    if (file.fail()) {
        ostringstream msg;
        msg << "Dictionary file \"" << dictionary_file << "\" could not be opened.";
        msg << " Check that the file exists." << endl;
        throw runtime_error(msg.str());
    }
    while (getline(file, line)) {
        if (!line.empty()) {
            if (line.length() != 1) {
                ostringstream msg;
                msg << "Each line of dictionary file should have 1 character." << endl;
                msg << "Read line " << line << ", with length " << line.length() << "." << endl;
                throw runtime_error(msg.str());
            }
            char c = tolower(line[0]);
            if (dictmap.find(c) == dictmap.end()) {
                dictmap[c] = key_val;
                key_val++;
            }
        }
    }
    file.close();
    return dictmap;
}

// infer dictionary from the training file
map<char, int> infer_dict(const string train_file) {
    ifstream file;
    string line;
    bool is_label = true;
    map<char, int> dictmap;
    int key_val = 1;
    int length;

    file.open(train_file);
    if (file.fail()) {
        ostringstream msg;
        msg << "Training file \"" << train_file << "\" could not be opened.";
        msg << " Check that the file exists." << endl;
        throw runtime_error(msg.str());
    }
    while (getline(file, line)) {
        trim_line(line);
        if (!line.empty()) {
            if (is_label) {
                is_label = false;
            } else {
                length = line.length();
                for (int i = 0; i < length; i++){
                    if (dictmap[tolower(line[i])] == 0) {
                        dictmap[tolower(line[i])] = key_val;
                        key_val++;
                    }
                }
                is_label = true;
            }
        }
    }
    file.close();
    return dictmap;
}

DataReader::DataReader(const string train_file, const string dictionary_file) {
    if (dictionary_file.empty()) {
        this->dictmap = infer_dict(train_file);
    } else {
        this->dictmap = read_dict(dictionary_file);
    }
}

void DataReader::read_data(const string data_file, bool train) {
    vector<vector<int> > sequences;
    vector<int> labels;
    int num_str = 0;
    int length = 0;
    int label_as_num;
    bool is_label = true;
    string line, label;
    ifstream file;

    file.open(data_file);
    if (file.fail()) {
        ostringstream msg;
        msg << "Data file \"" << data_file << "\" could not be opened." << endl;
        throw runtime_error(msg.str());
    }

    while (getline(file, line)) {
        trim_line(line);
        if (!line.empty()) {
            if (is_label) {
                string::size_type pos = line.find_first_of('>');
                label = line.substr(pos + 1);
                if (label.length() > 2) labelErrorAndExit(label);
                label_as_num = stoi(label) == 0 ? -1 : stoi(label);
                labels.push_back(label_as_num);
                is_label = false;
            } else {
                if (length > STRMAXLEN) {
                    line = line.substr(0, STRMAXLEN);
                }
                length = line.length();

                vector<int> seq;
                for (int i = 0; i < length; i++) {
                    seq.push_back(this->dictmap[tolower(line[i])]);
                }
                sequences.push_back(seq);
                num_str++;
                is_label = true;
            }
        }
    }

    cout << "Read " << num_str << " sequences from \"" << data_file << "\"" << endl;

    if (train) {
        this->train_seq = sequences;
        this->train_labels = labels;
        this->num_train_str = num_str;
    } else {
        this->test_seq = sequences;
        this->test_labels = labels;
        this->num_test_str = num_str;
    }
    this->total_num_str += num_str;
}
