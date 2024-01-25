#pragma once

#include "neuron.h"

/*
 * ACTIVATION FUNCTIONS
 */
enum ActivationType {
    NONE,
    SIGMOID,
    RELU
};

float activate(float _x, ActivationType _type);
float activateDer(float _x, ActivationType _type);

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
