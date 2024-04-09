#pragma once

#include "layer.h"

const string WEIGHTS_PATH = "C:/Users/semio/Downloads/weights.net";

const int INPUT_NEURONS = 12 * 64;
const int HIDDEN_NEURONS1 = 512;
const int HIDDEN_NEURONS2 = 64;
const int OUTPUT_NEURONS = 1;

const float LEARNING_RATE = 0.001f;

class Network {
public:
    explicit Network(bool loadWeights = false);
    ~Network();

    float feedForward(SparseInput &sparseInput);
    void feedBackward(float target);

    float evaluate(string &fen);

    void save();
    void load();

    void train(vector<SparseInput> &data, int epochs, int batchSize);

private:
    Layer **layers;
    int numLayers;

    float cost(const vector<SparseInput> &data);

};

inline float Network::cost(const vector<SparseInput> &data) {
    float totalCost = 0;
    for (auto d: data) {
        float prediction = feedForward(d);
        float error = prediction - d.target;
        totalCost += error * error;
    }
    return totalCost / data.size();
}
