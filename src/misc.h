#ifndef ASTRA_NNETWORK_MISC_H
#define ASTRA_NNETWORK_MISC_H

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


#endif //ASTRA_NNETWORK_MISC_H
