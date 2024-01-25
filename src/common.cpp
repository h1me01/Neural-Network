#include "common.h"

namespace Tools {
    random_device rd;
    mt19937 gen(rd());
}

float getBit(uint64_t bitboard, int pos) {
    uint64_t maskedBit = (bitboard >> pos) & 1;
    return static_cast<float>(maskedBit);
}

void shuffleData(vector<NetInput>& data) {
    for (int i = data.size() - 1; i > 0; --i) {
        uniform_int_distribution<int> dist(0, i);
        int j = dist(Tools::gen);
        swap(data[i], data[j]);
    }
}

vector<NetInput> loadNetData(const string &filePath, int dataSize) {
    vector<NetInput> netData;

    ifstream file(filePath, ios::binary);
    if (!file.is_open()) {
        cerr << "Error: Unable to open file for reading." << endl;
        return netData;
    }

    int count = 0;
    NetInput input;
    while (file.read(reinterpret_cast<char *>(&input), sizeof(NetInput))) {
        if (dataSize <= count) {
            break;
        }
        count++;
        netData.push_back(input);
    }

    file.close();
    return netData;
}

int pieceIndex(char _c) {
    const char pieces[] = "PpNnBbRrQqKk";
    for (int i = 0; pieces[i]; ++i) {
        if (_c == pieces[i]) {
            return i;
        }
    }
    return -1;
}

vector<float> fenToInput(string &fen) {
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

vector<float> normalizeTargets(vector<float> &targetValues) {
    const float minValue = -125;
    const float maxValue = 125;

    vector<float> normalizedValues;
    for (float value: targetValues) {
        float normalizedValue = (value - minValue) / (maxValue - minValue);
        normalizedValues.push_back(normalizedValue);
    }
    return normalizedValues;
}
