#include "activation.h"

#include <cmath>

float relu(float x) {
    return (x > 0) ? x : 0;
}

float leakyRelu(float x) {
    return (x > 0) ? x : 0.01f * x;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float reluDer(float x) {
    return (x > 0) ? 1.0f : 0;
}

float leakyReluDer(float x) {
    return (x > 0) ? 1.0f : 0.01f;
}

float sigmoidDer(float x) {
    float sig = sigmoid(x);
    return sig * (1.0f - sig);
}

float activate(float x, ActivationType type) {
    switch (type) {
        case RELU:       return relu(x);
        case LEAKY_RELU: return leakyRelu(x);
        case SIGMOID:    return sigmoid(x);
        default:         return 0;
    }
}

float activateDer(float x, ActivationType type) {
    switch (type) {
        case RELU:       return reluDer(x);
        case LEAKY_RELU: return leakyReluDer(x);
        case SIGMOID:    return sigmoidDer(x);
        default:         return 0;
    }
}
