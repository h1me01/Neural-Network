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

struct NetInput {
    uint64_t pieces[NUM_COLORS][6]{};
    float target;

    NetInput() {
        target = 0.0f;
    }
};

namespace Tools {
    inline random_device rd;
    inline mt19937 gen(rd());
} // namespace Tools

inline float getBit(uint64_t b, int pos) {
    return (b >> pos) & 1;
}

inline void shuffleData(vector<NetInput>& data) {
    uniform_int_distribution<int> dist;
    for (int i = data.size() - 1; i > 0; --i) {
        swap(data[i], data[dist(Tools::gen, decltype(dist)::param_type{0, i})]);
    }
}

inline vector<NetInput> loadNetData(const string &filePath, int dataSize = INT_MAX) {
    vector<NetInput> netData;
    ifstream file(filePath, ios::binary);

    if (!file) {
        cerr << "Error: Unable to open file for reading.\n";
        return netData;
    }

    NetInput input;
    while (netData.size() < dataSize && file.read(reinterpret_cast<char *>(&input), sizeof(NetInput))) {
        netData.push_back(input);
    }

    return netData;
}

inline int pieceIndex(char _c) {
    const string pieces = "PpNnBbRrQqKk";
    return pieces.find(_c);
}

inline vector<float> fenToInput(string &fen) {
    vector<float> input(768, 0);

    int rank = 7, file = -1;
    for (char c: fen) {
        if (c == ' ') {
            break;
        }

        if (c == '/') {
            rank--;
            file = -1;
        } else if (isdigit(c)) {
            file += (c - '0');
        } else {
            file++;
            input[64 * pieceIndex(c) + (8 * rank + file)] = 1;
        }
    }

    return input;
}

inline vector<float> normalizeTargets(vector<float> &targetValues) {
    const float minValue = -125;
    const float maxValue = 125;

    vector<float> normalizedValues;
    for (float value: targetValues) {
        float normalizedValue = (value - minValue) / (maxValue - minValue);
        normalizedValues.push_back(normalizedValue);
    }
    return normalizedValues;
}
