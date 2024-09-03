#ifndef ASTRA_NNETWORK_LAYER_H
#define ASTRA_NNETWORK_LAYER_H

#include "activation.h"
#include "neuron.h"

class Layer {
public:
    Layer(int input_size, int output_size, ActivationType activation_type) :
    input_size(input_size), output_size(output_size), activation_type(activation_type) {
        neurons = vector<Neuron*>(output_size);
        activations = new float[output_size];
        deltas = new float[output_size];
        input = new float[input_size];

        for (int i = 0; i < output_size; i++) {
            neurons[i] = new Neuron(input_size);
        }
    }

    ~Layer() {
        for (int i = 0; i < output_size; i++) {
            delete neurons[i];
        }
        delete[] activations;
        delete[] deltas;
        delete[] input;
    }

    void feedForward(float *current_input) const {
        copy_n(current_input, input_size, input);

        for (int i = 0; i < output_size; i++) {
            float dot = neurons[i]->dotProduct(input);
            activations[i] = activate(dot, activation_type);
        }
    }

    void calcDeltas(const Layer *next_layer, float target = 0) const {
        if(next_layer == nullptr) {
            // update output layer
            // use mean squared error
            deltas[0] = 2 * (activations[0] - target) * activateDer(activations[0], activation_type);
        } else {
            // update hidden layer(s)
            for (int i = 0; i < output_size; i++) {
                float delta = 0;
                for (int j = 0; j < next_layer->output_size; j++) {
                    float next_layer_weight = next_layer->neurons[j]->getWeights()[i];
                    delta += next_layer_weight * next_layer->deltas[j];
                }

                delta *= activateDer(activations[i], activation_type);
                deltas[i] = delta;
            }
        }
    }

    void updateGradients() const {
        for (int i = 0; i < output_size; i++) {
            __m512 delta_avx = _mm512_set1_ps(deltas[i]);
            float *weight_graidents = neurons[i]->getWeightGrads();

            for (int j = 0; j + 15 < input_size; j += 16) {
                __m512 input_avx = _mm512_loadu_ps(input + j);
                __m512 current_grad_avx = _mm512_loadu_ps(weight_graidents + j);
                __m512 inputWeightDer_avx = _mm512_fmadd_ps(input_avx, delta_avx, current_grad_avx);
                _mm512_storeu_ps(weight_graidents + j, inputWeightDer_avx);
            }

            neurons[i]->updateBiasGrad(deltas[i]);
        }
    }

    void updateNeurons(const float lr) const {
        for (int i = 0; i < output_size; i++) {
            neurons[i]->update(lr);
        }
    }

    void clearAllGradients() const {
        for (int i = 0; i < output_size; i++) {
            neurons[i]->clearGradients();
        }
    }

    [[nodiscard]] float* getActivations() const {
        // don't return the actual pointer of activations
        // since it might get deallocated
        auto* temp = new float[output_size];
        copy_n(activations, output_size, temp);
        return temp;
    }

    [[nodiscard]] vector<Neuron*> getNeurons() const {
        return neurons;
    }

    [[nodiscard]] int getInputSize() const {
        return input_size;
    }

    [[nodiscard]] int getOutputSize() const {
        return output_size;
    }

private:
    ActivationType activation_type;
    int input_size;
    int output_size;
    vector<Neuron*> neurons;
    float *activations;
    float *deltas;
    float *input;

};

#endif //ASTRA_NNETWORK_LAYER_H
