#ifndef ASTRA_NNETWORK_ACTIVATION_H
#define ASTRA_NNETWORK_ACTIVATION_H

#include <cmath>

enum ActivationType {
    NONE,
    RELU,
    LEAKY_RELU,
    SIGMOID
};

float relu(float x);
float leakyRelu(float x);
float sigmoid(float x);

float reluDer(float x);
float leakyReluDer(float x) ;
float sigmoidDer(float x);

float activate(float x, ActivationType type);
float activateDer(float x, ActivationType type);


#endif //ASTRA_NNETWORK_ACTIVATION_H
