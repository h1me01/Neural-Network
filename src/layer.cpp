#include "layer.h"

/*
 * ACTIVATION
 */
float activate(float _x, ActivationType _type) {
    switch (_type) {
        case SIGMOID:
            return 1.0f / (1.0f + exp(-_x));
        case RELU:
            return (_x > 0) ? _x : 0;
        default:
            return 0;
    }
}

float activateDer(float _x, ActivationType _type) {
    switch (_type) {
        case SIGMOID:
            return activate(_x, SIGMOID) * (1.0f - activate(_x, SIGMOID));
        case RELU:
            return (_x > 0) ? 1.0f : 0;
        default:
            return 0;
    }
}

/*
 * LAYER CLASS
 */
Layer::Layer(int numPrevNeurons, int numNeurons, ActivationType activationType) :
        numPrevNeurons(numPrevNeurons), numNeurons(numNeurons), activationType(activationType) {
    neurons = new Neuron *[numNeurons];
    weightedInputs = new float[numNeurons];
    activations = new float[numNeurons];
    input = new float[numPrevNeurons];

    for (int i = 0; i < numNeurons; ++i) {
        neurons[i] = new Neuron(numPrevNeurons);
        weightedInputs[i] = 0;
        activations[i] = 0;
    }
}

Layer::~Layer() {
    for (int i = 0; i < numNeurons; ++i) {
        delete neurons[i];
    }
    delete[] neurons;
    delete[] weightedInputs;
    delete[] activations;
    delete[] input;
}

float *Layer::feedForward(float *currentInput) {
    copy_n(currentInput, numPrevNeurons, input);

    for (int i = 0; i < numNeurons; ++i) {
        weightedInputs[i] = neurons[i]->dotProduct(input);
        activations[i] = activate(weightedInputs[i], activationType);
    }
    return activations;
}

float Layer::calcOutputDelta(float target) {
    return 2 * (activations[0] - target) * activateDer(weightedInputs[0], SIGMOID);
}

float *Layer::calcHiddenDeltas(Layer *prevLayer, float *prevDeltas) {
    float *deltas = new float[numNeurons];
    for (int i = 0; i < numNeurons; ++i) {
        float delta = 0;
        for (int j = 0; j < prevLayer->numNeurons; ++j) {
            float prevLayerWeight = prevLayer->neurons[j]->getWeights()[i];
            delta += prevLayerWeight * prevDeltas[j];
        }
        delta *= activateDer(weightedInputs[i], activationType);
        deltas[i] = delta;
    }
    return deltas;
}

void Layer::updateNeurons(float lr) {
    for (int i = 0; i < numNeurons; ++i) {
        neurons[i]->update(lr);
    }
}

void Layer::updateGradients(float *deltas) {
    for (int i = 0; i < numNeurons; ++i) {
        __m256 delta_avx = _mm256_set1_ps(deltas[i]);

        for (int j = 0; j < numPrevNeurons; j += 8) {
            __m256 input_avx = _mm256_loadu_ps(input + j);
            __m256 inputWeightDer_avx = _mm256_mul_ps(input_avx, delta_avx);

            float elements[8];
            _mm256_storeu_ps(elements, inputWeightDer_avx);

            neurons[i]->updateGradientWeight(j, elements[0]);
            neurons[i]->updateGradientWeight(j + 1, elements[1]);
            neurons[i]->updateGradientWeight(j + 2, elements[2]);
            neurons[i]->updateGradientWeight(j + 3, elements[3]);
            neurons[i]->updateGradientWeight(j + 4, elements[4]);
            neurons[i]->updateGradientWeight(j + 5, elements[5]);
            neurons[i]->updateGradientWeight(j + 6, elements[6]);
            neurons[i]->updateGradientWeight(j + 7, elements[7]);
        }

        neurons[i]->updateGradientBias(deltas[i]);
    }
}

void Layer::clearAllGradients() {
    for (int i = 0; i < numNeurons; ++i) {
        neurons[i]->clearGradients();
    }
}
