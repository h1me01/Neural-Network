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

    NetInput() {
        target = 0.0f;
    }
};

void shuffleData(vector<NetInput> &data);

vector<NetInput> getNetData(const string &filePath, int dataSize = INT_MAX);

float* getSparseInput(NetInput &netInput);

vector<float> fenToInput(string &fen);

vector<float> normalizeTargets(vector<float> &targetValues, float minValue = -125, float maxValue = 125);

#endif //ASTRA_NNETWORK_DATASET_H
