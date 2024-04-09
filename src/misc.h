#pragma once

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <array>
#include <cmath>
#include <random>
#include <algorithm>
#include <immintrin.h>
#include <cassert>

using namespace std;

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
    array<float, 12 * 64> input;
    float target;

    SparseInput() : input{}, target(0) {}

    void set(int idx) {
        assert(idx >= 0 && idx < (12 * 64));
        input[idx] = true;
    }

    float get(int idx) {
        assert(idx >= 0 && idx < (12 * 64));
        return input[idx];
    }
};

namespace Tools {
    inline random_device rd;
    inline mt19937 gen(rd());
} // namespace Tools

const int DEBRUIJN64[64] = {
        0, 47, 1, 56, 48, 27, 2, 60,
        57, 49, 41, 37, 28, 16, 3, 61,
        54, 58, 35, 52, 50, 42, 21, 44,
        38, 32, 29, 23, 17, 11, 4, 62,
        46, 55, 26, 59, 40, 36, 15, 53,
        34, 51, 20, 43, 31, 22, 10, 45,
        25, 39, 14, 33, 19, 30, 9, 24,
        13, 18, 8, 12, 7, 6, 5, 63
};

constexpr int bsf(uint64_t b) {
    return DEBRUIJN64[0x03f79d71b4cb0a89 * (b ^ (b - 1)) >> 58];
}

inline int popLsb(uint64_t &b) {
    int lsb = bsf(b);
    b &= b - 1;
    return lsb;
}

inline void shuffleData(vector<SparseInput> &data) {
    uniform_int_distribution<int> dist;
    for (int i = data.size() - 1; i > 0; --i) {
        swap(data[i], data[dist(Tools::gen, decltype(dist)::param_type{0, i})]);
    }
}

inline int pieceIndex(char c) {
    const string pieces = "pnbrwk";
    return pieces.find(tolower(c));
}

inline int mirrorVertically(int sq) {
    int rank = sq / 8;
    int file = sq % 8;
    
    return 8 * (7 - rank) + file;
}

inline int index(int psq, char p, Color view) {
    if (view != WHITE) {
        psq = mirrorVertically(psq);
    }

    Color pc = isupper(p) ? WHITE : BLACK;
    return psq + 64 * pieceIndex(p) + (pc != view) * 64 * 6;
}

inline int index(int psq, int pt, Color pc, Color view) {
    if (view != WHITE) {
        psq = mirrorVertically(psq);
    }

    return psq + 64 * pt + (pc != view) * 64 * 6;
}

inline SparseInput getSparseInput(NetInput &netInput) {
    SparseInput input;
    input.target = netInput.target;

    for (int i = 0; i < NUM_COLORS; ++i) {
        for (int j = 0; j < 6; ++j) {
            uint64_t piece = netInput.pieces[i][j];
            while (piece) {
                int sq = popLsb(piece);
                int idx = index(sq, j, (Color) i, netInput.stm);

                input.set(idx);
            }
        }
    }

    return input;
}

inline vector<SparseInput> getSparseData(const string &filePath, int dataSize = INT_MAX) {
    vector<SparseInput> sparseData;
    ifstream file(filePath, ios::binary);

    if (!file) {
        cerr << "Error: Unable to open file for reading.\n";
        return sparseData;
    }

    NetInput netInput;
    while (sparseData.size() < dataSize && file.read(reinterpret_cast<char *>(&netInput), sizeof(NetInput))) {
        sparseData.push_back(getSparseInput(netInput));
    }

    return sparseData;
}

inline SparseInput fenToInput(string &fen) {
    SparseInput input;
    Color stm = fen.find('w') != string::npos ? WHITE : BLACK;

    int rank = 7, file = -1;
    for (char c: fen) {
        if (c == ' ') { break; }
        if (c == '/') {
            rank--;
            file = -1;
        } else if (isdigit(c)) {
            file += (c - '0');
        } else {
            file++;
            int sq = 8 * rank + file;
            int idx = index(sq, c, stm);
            input.set(idx);
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
