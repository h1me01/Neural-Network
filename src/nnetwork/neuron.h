#ifndef ASTRA_NNETWORK_NEURON_H
#define ASTRA_NNETWORK_NEURON_H

#include "adam.h"

/*
 * HE INIT
 */
float heInit(int numPrevNeurons);

/*
 * NEURON CLASS
 */
class Neuron {
public:
    Neuron(int numPrevNeurons);

    ~Neuron();

    float dotProduct(const float *input);

    void clearGradients();

    void update(float lr);

    void updateGradientWeight(int i, float val) {
        gradientWeights[i] += val;
    }

    void updateGradientBias(float val) {
        gradientBias += val;
    }

    int getNumWeights() {
        return numWeights;
    }

    float *getWeights() const {
        return weights;
    }

    void setWeights(float *newWeights) {
        delete[] weights;
        weights = newWeights;
    }

    float getBias() const {
        return bias;
    }

    void setBias(float newBias) {
        bias = newBias;
    }

private:
    Adam adam;

    int numWeights;

    float *gradientWeights;
    float gradientBias;
    float *weights;
    float bias;

};

#endif //ASTRA_NNETWORK_NEURON_H
