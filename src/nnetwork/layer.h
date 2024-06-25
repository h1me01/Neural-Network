#ifndef ASTRA_NNETWORK_LAYER_H
#define ASTRA_NNETWORK_LAYER_H

#include "activation.h"
#include "neuron.h"

class Layer {
public:
    Layer(int numPrevNeurons, int numNeurons, ActivationType activationType) :
    numPrevNeurons(numPrevNeurons), numNeurons(numNeurons), activationType(activationType) {
        neurons = new Neuron *[numNeurons];
        weightedInputs = new float[numNeurons];
        activations = new float[numNeurons];
        deltas = new float[numNeurons];
        input = new float[numPrevNeurons];

        for (int i = 0; i < numNeurons; ++i)
            neurons[i] = new Neuron(numPrevNeurons);
    }

    ~Layer() {
        for (int i = 0; i < numNeurons; ++i)
            delete neurons[i];

        delete[] neurons;
        delete[] weightedInputs;
        delete[] activations;
        delete[] deltas;
        delete[] input;
    }

    float *feedForward(float *currentInput) const {
        copy_n(currentInput, numPrevNeurons, input);

        for (int i = 0; i < numNeurons; ++i) {
            weightedInputs[i] = neurons[i]->dotProduct(input);
            activations[i] = activate(weightedInputs[i], activationType);
        }

        return activations;
    }

    void calcOutputDelta(float target) const {
        deltas[0] =  2 * (activations[0] - target) * activateDer(weightedInputs[0], SIGMOID);
    }

    void calcHiddenDeltas(Layer *prevLayer) const {
        for (int i = 0; i < numNeurons; ++i) {
            float delta = 0;
            for (int j = 0; j < prevLayer->numNeurons; ++j) {
                float prevLayerWeight = prevLayer->neurons[j]->getWeights()[i];
                delta += prevLayerWeight * prevLayer->deltas[j];
            }

            delta *= activateDer(weightedInputs[i], activationType);
            deltas[i] = delta;
        }
    }

    void updateNeurons(float lr) const {
        for (int i = 0; i < numNeurons; ++i)
            neurons[i]->update(lr);
    }

    void updateGradients() const {
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

    void clearAllGradients() const {
        for (int i = 0; i < numNeurons; ++i)
            neurons[i]->clearGradients();
    }

    Neuron **getNeurons() const { return neurons; }
    int getNumPrevNeurons() const { return numPrevNeurons; }
    int getNumNeurons() const { return numNeurons; }

private:
    ActivationType activationType;

    int numPrevNeurons;
    int numNeurons;

    Neuron **neurons;

    float *weightedInputs;
    float *activations;
    float *deltas;
    float *input;
};

#endif //ASTRA_NNETWORK_LAYER_H
