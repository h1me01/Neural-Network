#ifndef ASTRA_NNETWORK_ACTIVATION_H
#define ASTRA_NNETWORK_ACTIVATION_H

#include <cmath>

enum ActivationType {
    NONE,
    RELU,
    LEAKY_RELU,
    SIGMOID
};

inline float relu(float x) { return x > 0 ? x : 0; }
inline float leakyRelu(float x) { return x > 0 ? x : 0.01f * x; }
inline float sigmoid(float x) { return 1 / (1 + std::exp(-x)); }

inline float reluDer(float x) { return x > 0 ? 1 : 0; }
inline float leakyReluDer(float x) { return x > 0 ? 1 : 0.01f; }

inline float sigmoidDer(float x) {
    float sig = sigmoid(x);
    return sig * (1 - sig);
}

inline float activate(float x, ActivationType type) {
    switch (type) {
        case RELU:
            return relu(x);
        case LEAKY_RELU:
            return leakyRelu(x);
        case SIGMOID:
            return sigmoid(x);
        default:
            return x;
    }
}

inline float activateDer(float x, ActivationType type) {
    switch (type) {
        case RELU:
            return reluDer(x);
        case LEAKY_RELU:
            return leakyReluDer(x);
        case SIGMOID:
            return sigmoidDer(x);
        default:
            return 1;
    }
}

#endif //ASTRA_NNETWORK_ACTIVATION_H
