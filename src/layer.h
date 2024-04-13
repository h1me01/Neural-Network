#ifndef ASTRA_NNETWORK_LAYER_H
#define ASTRA_NNETWORK_LAYER_H

#include "neuron.h"

/*
 * ACTIVATION FUNCTIONS
 */
enum ActivationType {
    NONE,
    SIGMOID,
    LEAKY_RELU,
    RELU
};

float activate(float x, ActivationType type);
float activateDer(float x, ActivationType type);

/*
 * LAYER CLASS
 */
class Layer {
public:
    Layer(int numPrevNeurons, int numNeurons, ActivationType activationType);
    ~Layer();

    float *feedForward(float *currentInput);

    float calcOutputDelta(float target);
    float *calcHiddenDeltas(Layer *prevLayer, float *prevDeltas);

    void updateNeurons(float lr);
    void updateGradients(float *deltas);

    void clearAllGradients();

    Neuron **getNeurons() {
        return neurons;
    }

    int getNumPrevNeurons() {
        return numPrevNeurons;
    }

    int getNumNeurons() {
        return numNeurons;
    }

private:
    ActivationType activationType;
    int numPrevNeurons;
    int numNeurons;
    Neuron **neurons;
    float *weightedInputs;
    float *activations;
    float *input;

};

#endif //ASTRA_NNETWORK_LAYER_H
