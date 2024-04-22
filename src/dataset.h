#ifndef ASTRA_NNETWORK_DATASET_H
#define ASTRA_NNETWORK_DATASET_H

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <immintrin.h>
#include <cassert>
#include "misc.h"

#define NUM_FEATURES 12 * 64

enum Color : int {
    WHITE, BLACK, NUM_COLORS = 2
};

struct NetInput {
    uint64_t pieces[NUM_COLORS][6]{};
    float target;
    Color stm;

    NetInput() {
        target = 0.0f;
    }
};

struct SparseInput {
    float input[NUM_FEATURES]{};
    float target;

    SparseInput() : target(0) {}

    void set(int idx) {
        assert(idx >= 0 && idx < NUM_FEATURES);
        input[idx] = 1;
    }
};

void shuffleData(vector<SparseInput> &data);

int pieceIndex(char c);

int mirrorVertically(int sq);

int index(int psq, char p, Color view);
int index(int psq, int pt, Color pc, Color view);

SparseInput getSparseInput(NetInput &netInput);

vector<SparseInput> getSparseData(const string &filePath, int dataSize = INT_MAX);

SparseInput fenToInput(string &fen);

vector<float> normalizeTargets(vector<float> &targetValues);

#endif //ASTRA_NNETWORK_DATASET_H
