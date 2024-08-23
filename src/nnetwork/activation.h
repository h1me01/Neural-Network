#ifndef ASTRA_NNETWORK_ACTIVATION_H
#define ASTRA_NNETWORK_ACTIVATION_H

#include <cmath>

enum ActivationType {
    NONE,
    RELU,
    SIGMOID
};

constexpr float SigmoidScaler = 0.00055254;

inline float relu(float x) { return x > 0 ? x : 0; }
inline float sigmoid(float x) { return 1 / (1 + std::exp(-SigmoidScaler * x)); }

inline float reluDer(float x) { return x > 0 ? 1 : 0; }
inline float sigmoidDer(float x) { return x * (1 - x) * SigmoidScaler; }

inline float activate(float x, ActivationType type) {
    switch (type) {
        case RELU:
            return relu(x);
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
        case SIGMOID:
            return sigmoidDer(x);
        default:
            return 0;
    }
}

#endif //ASTRA_NNETWORK_ACTIVATION_H
