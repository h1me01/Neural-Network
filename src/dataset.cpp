#include "dataset.h"

/*
 * HELPER FUNCTIONS
 */
int pieceType(char c) {
    const string pieces = "pnbrqk";
    return pieces.find(tolower(c));
}

int mirrorVertically(int sq) {
    return sq ^ 56;
}

int index(int psq, char p, Color view) {
    Color pc = isupper(p) ? WHITE : BLACK;
    if(view != WHITE) {
        psq = mirrorVertically(psq);
    }

    return psq + pieceType(p) * 64 + (pc != view) * 64 * 6;
}

int index(int psq, int pt, Color pc, Color view) {
    if(view != WHITE) {
        psq = mirrorVertically(psq);
    }

    return psq + pt * 64 + (pc != view) * 64 * 6;
}

/*
 * PUBLIC FUNCTIONS
 */
void shuffleData(vector<NetInput> &data) {
    for (size_t i = data.size() - 1; i > 0; --i) {
        uniform_int_distribution<size_t> dis(0, i);
        size_t j = dis(Tools::gen);
        swap(data[i], data[j]);
    }
}

vector<NetInput> getNetData(const string &path, int data_size) {
    vector<NetInput> net_data;
    ifstream file(path, ios::binary);

    if (!file) {
        cerr << "Error: Unable to open file for reading.\n";
        return net_data;
    }

    NetInput netInput;
    while (file.read(reinterpret_cast<char *>(&netInput), sizeof(NetInput))) {
        net_data.push_back(netInput);

        if(net_data.size() >= data_size)
            break;
    }

    return net_data;
}

float *getSparseInput(NetInput &net_input) {
    auto *sparse = new float[NUM_FEATURES]{};
    Color stm = net_input.stm;

    for (int i = 0; i < NUM_COLORS; ++i) {
        for (int j = 0; j < 6; ++j) {
            uint64_t piece = net_input.pieces[i][j];
            while (piece) {
                int sq = popLsb(piece);
                int idx = index(sq, j, (Color) i, stm);

                sparse[idx] = 1;
            }
        }
    }

    return sparse;
}

vector<float> fenToInput(string &fen) {
    vector<float> input(NUM_FEATURES, 0);
    Color stm = fen.find('w') != string::npos ? WHITE : BLACK;

    int rank = 7, file = -1;
    for (char c: fen) {
        if (c == ' ') break;
        if (c == '/') {
            rank--;
            file = -1;
        } else if (isdigit(c)) {
            file += c - '0';
        } else {
            file++;
            int sq = 8 * rank + file;
            int idx = index(sq, c, stm);

            input[idx] = 1.0f;
        }
    }

    return input;
}
