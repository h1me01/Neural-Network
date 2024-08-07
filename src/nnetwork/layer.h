#ifndef ASTRA_NNETWORK_LAYER_H
#define ASTRA_NNETWORK_LAYER_H

#include "activation.h"
#include "neuron.h"

class Layer {
public:
    Layer(int numFeatures, int numNeurons, ActivationType activationType) :
    numFeatures(numFeatures), numNeurons(numNeurons), activationType(activationType) {
        neurons = new Neuron *[numNeurons];
        weightedInputs = new float[numNeurons];
        activations = new float[numNeurons];
        deltas = new float[numNeurons];
        input = new float[numFeatures];

        for (int i = 0; i < numNeurons; ++i)
            neurons[i] = new Neuron(numFeatures);
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
        copy_n(currentInput, numFeatures, input);

        // only deallocate the input to the hidden layer
        // note that because we return the activations of the current layer
        // we don't want to deallocate them by accident, since they are used
        // in the next layer and if the next layer cant access them, the whole
        // program will crash
        if(numFeatures == NUM_FEATURES)
            delete[] currentInput;

        for (int i = 0; i < numNeurons; ++i) {
            weightedInputs[i] = neurons[i]->dotProduct(input);
            activations[i] = activate(weightedInputs[i], activationType);
        }

        return activations;
    }

    void calcDeltas(const Layer *prevLayer, float target = 0) const {
        if(prevLayer == nullptr) {
            // update output layer
            deltas[0] = 2 * (activations[0] - target) * activateDer(weightedInputs[0], activationType);
        } else {
            // update hidden layer(s)
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
    }

    void updateGradients() const {
        for (int i = 0; i < numNeurons; ++i) {
            __m512 delta_avx = _mm512_set1_ps(deltas[i]);
            float *gradientWeights = neurons[i]->getGradientWeights();

            for (int j = 0; j + 15 < numFeatures; j += 16) {
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
        return numFeatures;
    }

    [[nodiscard]] int getNumNeurons() const {
        return numNeurons;
    }

private:
    ActivationType activationType;

    int numFeatures;
    int numNeurons;

    Neuron **neurons;

    float *weightedInputs;
    float *activations;
    float *deltas;
    float *input;
};

#endif //ASTRA_NNETWORK_LAYER_H
