#include "network.h"

const string DATA_PATH = "C:/Users/semio/Downloads/data.bin";

const int EPOCHS = 10;
const int BATCH_SIZE = 32;

vector<string> TEST_POSITIONS = {
        "3k4/5K2/8/4P3/8/8/8/8 b - - 2 11",
        "4k3/8/3KP3/8/8/8/8/8 b - - 0 10",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "2k2bnr/ppp2ppp/2n5/3rP3/4pB2/8/PPP1KPPP/R5NR w - - 0 10",
        "1k5r/p1pR3p/2p3pb/4Pp2/4p3/8/PPP1KPPP/7R b - - 1 16",
        "7Q/6K1/p7/2p1Q3/8/8/P1P5/2k5 b - - 4 57",
        "r5k1/p5p1/4p2p/4p3/3p4/P6K/5q2/8 b - - 0 33"
};

vector<float> TEST_POSITIONS_EVAL = {
        114, 0, 0.2f, -5.5f, 4.1f, 122, -122
};

int main() {
    /*
     * LOAD AND NORMALIZE DATA
     */
    vector<SparseInput> DATA = getSparseData(DATA_PATH);

    vector<float> targets;
    for (auto &d: DATA) {
        targets.push_back(d.target);
    }

    vector<float> targetsNormalized = normalizeTargets(targets);
    for (int i = 0; i < DATA.size(); ++i) {
        DATA[i].target = targetsNormalized[i];
    }

    /*
     * TRAIN NETWORK
     */
    Network net;
    net.train(DATA, EPOCHS, BATCH_SIZE);

    /*
     * PREDICTIONS
     */
    for (int i = 0; i < TEST_POSITIONS.size(); ++i) {
        float prediction = net.evaluate(TEST_POSITIONS[i]);
        cout << "Fen: " << TEST_POSITIONS[i] << endl;
        cout << "Prediction: " << prediction << endl;
        cout << "Target: " << TEST_POSITIONS_EVAL[i] << endl;
        cout << "--------------------------------------" << endl;
    }

    return 0;
}
