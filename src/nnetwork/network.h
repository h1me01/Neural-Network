#ifndef ASTRA_NNETWORK_NETWORK_H
#define ASTRA_NNETWORK_NETWORK_H

#include "layer.h"

const string WEIGHTS_PATH = "C:/Users/semio/Downloads/astra_weights_768-64-1.nnue";

const int INPUT_NEURONS = NUM_FEATURES;
const int HIDDEN_NEURONS1 = 64;
const int OUTPUT_NEURONS = 1;

const float LEARNING_RATE = 0.001f;

class Network {
public:
    explicit Network(bool loadWeights = false);
    ~Network();

    float feedForward(NetInput& netInput) const;
    void feedBackward(float target) const;

    float evaluate(string &fen) const;

    void saveWeights() const;
    void loadWeights() const;

    void train(vector<NetInput> &data, int epochs, int batchSize);

private:
    int numLayers;

    Layer **layers;

    float getLoss(const vector<NetInput> &data);

};

inline Network::Network(bool loadWeights) {
    numLayers = 2;

    layers = new Layer *[numLayers];
    layers[0] = new Layer(INPUT_NEURONS, HIDDEN_NEURONS1, RELU);
    layers[1] = new Layer(HIDDEN_NEURONS1, OUTPUT_NEURONS, SIGMOID);

    if (loadWeights) loadWeights();
}

inline Network::~Network() {
    for (int i = 0; i < numLayers; ++i)
        delete layers[i];

    delete[] layers;
}

inline float Network::getLoss(const vector<NetInput> &data) {
    float totalCost = 0;

    for (auto d: data) {
        float prediction = feedForward(d);
        float error = prediction - d.target;
        totalCost += error * error;
    }

    return totalCost / data.size();
}

#endif //ASTRA_NNETWORK_NETWORK_H
