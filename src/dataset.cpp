#include "dataset.h"
#include "nnetwork/activation.h"

/*
 * HELPER FUNCTIONS
 */
int pieceType(char c) {
    const string pieces = "pnbrqk";
    return pieces.find(tolower(c));
}

int index(int psq, char p, Color view) {
    Color pc = isupper(p) ? WHITE : BLACK;
    if (view != WHITE) {
        psq = psq ^ 56;
    }

    return psq + pieceType(p) * 64 + (pc != view) * 64 * 6;
}

int index(int psq, int pt, Color pc, Color view) {
    if (view != WHITE) {
        psq = psq ^ 56;
    }
    return psq + pt * 64 + (pc != view) * 64 * 6;
}

/*
 * PUBLIC FUNCTIONS
 */
void shuffleData(vector<NetInput> &data) {
    for (size_t i = data.size() - 1; i > 0; i--) {
        uniform_int_distribution<size_t> dis(0, i);
        const size_t j = dis(Tools::gen);
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
        netInput.target = sigmoid(netInput.target);
        net_data.push_back(netInput);

        if (net_data.size() >= data_size) {
            break;
        }
    }

    return net_data;
}

float *getSparseInput(const NetInput &net_input) {
    auto *sparse = new float[2 * NUM_FEATURES]{};
    Color stm = net_input.stm;
    Color opp_stm = stm == WHITE ? BLACK : WHITE;

    for (const auto &i: net_input.pieces) {
        for (int j = 0; j < 6; j++) {
            uint64_t piece = i[j];
            while (piece) {
                const int sq = popLsb(piece);
                const int w_idx = index(sq, j, stm, stm);
                const int b_idx = index(sq, j, opp_stm, stm);
                sparse[w_idx] = 1;
                sparse[b_idx] = 1;
            }
        }
    }

    return sparse;
}

vector<float> fenToInput(string &fen) {
    vector<float> input(NUM_FEATURES, 0);
    Color stm = fen.find('w') != string::npos ? WHITE : BLACK;

    int rank = 7, file = -1;
    for (const char c: fen) {
        if (c == ' ') break;
        if (c == '/') {
            rank--;
            file = -1;
        } else if (isdigit(c)) {
            file += c - '0';
        } else {
            file++;
            const int sq = 8 * rank + file;
            const int idx = index(sq, c, stm);
            input[idx] = 1;
        }
    }

    return input;
}
