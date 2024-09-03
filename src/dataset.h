#ifndef ASTRA_NNETWORK_DATASET_H
#define ASTRA_NNETWORK_DATASET_H

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <cmath>
#include <immintrin.h>
#include "misc.h"

#define NUM_FEATURES (12 * 64)

enum Color : int {
    WHITE, BLACK, NUM_COLORS = 2
};

struct NetInput {
    uint64_t pieces[NUM_COLORS][6]{};
    float target;
    Color stm;
};

void shuffleData(vector<NetInput> &data);

vector<NetInput> getNetData(const string &path, int data_size = INT_MAX);

float* getSparseInput(const NetInput &net_input);

vector<float> fenToInput(string &fen);

#endif //ASTRA_NNETWORK_DATASET_H
