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

    void calcOutputDelta(const float target) const {
        deltas[0] =  2 * (activations[0] - target) * activateDer(weightedInputs[0], SIGMOID);
    }

    void calcHiddenDeltas(const Layer *prevLayer) const {
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

    void updateGradients() const {
        for (int i = 0; i < numNeurons; ++i) {
            __m512 delta_avx = _mm512_set1_ps(deltas[i]);
            float *gradientWeights = neurons[i]->getGradientWeights();

            for (int j = 0; j + 15 < numPrevNeurons; j += 16) {
                __m512 input_avx = _mm512_loadu_ps(input + j);
                __m512 current_grad_avx = _mm512_loadu_ps(gradientWeights + j);
                __m512 inputWeightDer_avx = _mm512_fmadd_ps(input_avx, delta_avx, current_grad_avx);
                _mm512_storeu_ps(gradientWeights + j, inputWeightDer_avx);
            }

            neurons[i]->updateGradientBias(deltas[i]);
        }
    }

    void updateNeurons(const float lr) const {
        for (int i = 0; i < numNeurons; ++i)
            neurons[i]->update(lr);
    }

    void clearAllGradients() const {
        for (int i = 0; i < numNeurons; ++i)
            neurons[i]->clearGradients();
    }

    [[nodiscard]] Neuron **getNeurons() const {
        return neurons;
    }

    [[nodiscard]] int getNumPrevNeurons() const {
        return numPrevNeurons;
    }

    [[nodiscard]] int getNumNeurons() const {
        return numNeurons;
    }

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
