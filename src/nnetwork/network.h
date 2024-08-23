#ifndef ASTRA_NNETWORK_NETWORK_H
#define ASTRA_NNETWORK_NETWORK_H

#include "layer.h"

constexpr int INPUT_SIZE = 768;
constexpr int HIDDEN_SIZE = 512;
constexpr int OUTPUT_SIZE = 1;

constexpr float LEARNING_RATE = 0.01f;

class Network {
public:
    explicit Network(const string &weights_path = "");
    ~Network();

    float feedForward(NetInput& net_input) const;
    void feedBackward(float target) const;

    float evaluate(string &fen) const;

    void saveWeights(int epoch) const;
    void loadWeights(const string& weights_path) const;

    void train(vector<NetInput> &data, int epochs, int batch_size);

private:
    int num_layers;
    Layer **layers;

    float getLoss(const vector<NetInput> &data);
};

inline Network::Network(const string& weights_path) {
    num_layers = 2;

    layers = new Layer *[num_layers];
    layers[0] = new Layer(INPUT_SIZE, HIDDEN_SIZE, RELU);
    layers[1] = new Layer(HIDDEN_SIZE, OUTPUT_SIZE, SIGMOID);

    if (!weights_path.empty())
        loadWeights(weights_path);
}

inline Network::~Network() {
    for (int i = 0; i < num_layers; ++i) {
        delete layers[i];
    }
    delete[] layers;
}

inline float Network::evaluate(string& fen) const {
    float* input = new float[NUM_FEATURES];
    vector<float> sparse_input = fenToInput(fen);
    copy(sparse_input.begin(), sparse_input.end(), input);

    for (int i = 0; i < num_layers; ++i) {
        input = layers[i]->feedForward(input);
    }

    constexpr float sigmoid_scalar = 0.00055254;
    return log(input[0] / (1 - input[0])) / sigmoid_scalar;
}

inline float Network::getLoss(const vector<NetInput> &data) {
    float loss = 0;
    for (auto d: data) {
        float prediction = feedForward(d);
        float error = prediction - d.target;
        loss += error * error;
    }

    return loss / data.size();
}

#endif //ASTRA_NNETWORK_NETWORK_H
