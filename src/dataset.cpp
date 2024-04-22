#include "dataset.h"

void shuffleData(vector<NetInput> &data) {
    uniform_int_distribution<int> dist;
    for (int i = data.size() - 1; i > 0; --i) {
        swap(data[i], data[dist(Tools::gen, decltype(dist)::param_type{0, i})]);
    }
}

int pieceIndex(char c) {
    const string pieces = "pnbrqk";
    return pieces.find(tolower(c));
}

int mirrorVertically(int sq) {
    return sq ^ 56;
}

int index(int psq, char p, Color view) {
    if (view != WHITE) {
        psq = mirrorVertically(psq);
    }

    Color pc = isupper(p) ? WHITE : BLACK;
    return psq + 64 * pieceIndex(p) + (pc != view) * 64 * 6;
}

int index(int psq, int pt, Color pc, Color view) {
    if (view != WHITE) {
        psq = mirrorVertically(psq);
    }

    return psq + 64 * pt + (pc != view) * 64 * 6;
}

float *getSparseInput(NetInput &netInput) {
    float *sparseInput = new float[NUM_FEATURES]{};

    for (int i = 0; i < NUM_COLORS; ++i) {
        for (int j = 0; j < 6; ++j) {
            uint64_t piece = netInput.pieces[i][j];
            while (piece) {
                int sq = popLsb(piece);
                int idx = index(sq, j, (Color) i, netInput.stm);

                sparseInput[idx] = 1.0f;
            }
        }
    }

    return sparseInput;
}

vector<NetInput> getSparseData(const string &filePath, int dataSize) {
    vector<NetInput> sparseData;
    ifstream file(filePath, ios::binary);

    if (!file) {
        cerr << "Error: Unable to open file for reading.\n";
        return sparseData;
    }

    NetInput netInput;
    while (sparseData.size() < dataSize && file.read(reinterpret_cast<char *>(&netInput), sizeof(NetInput))) {
        sparseData.push_back(netInput);
    }

    return sparseData;
}

vector<float> fenToInput(string &fen) {
    vector<float> input(NUM_FEATURES, 0);
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

            input[idx] = 1;
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
