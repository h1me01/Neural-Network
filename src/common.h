#pragma once

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <immintrin.h>

using namespace std;

enum Color : int {
    WHITE, BLACK, NUM_COLORS = 2
};

namespace Tools {
    extern random_device rd;
    extern mt19937 gen;
}

struct NetInput {
    uint64_t pieces[NUM_COLORS][6]{};
    float target;

    NetInput() {
        target = 0.0f;
    }
};

float getBit(uint64_t bitboard, int pos);

void shuffleData(vector<NetInput>& data);

vector<NetInput> loadNetData(const string &filePath, int dataSize = INT_MAX);

int pieceIndex(char _c);
vector<float> fenToInput(string &fen);

vector<float> normalizeTargets(vector<float> &targetValues);
