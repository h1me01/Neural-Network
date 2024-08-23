#ifndef ASTRA_NNETWORK_NEURON_H
#define ASTRA_NNETWORK_NEURON_H

#include "adam.h"

/*
 * HE INIT
 */
inline float heInit(int input_size) {
    normal_distribution<float> dist(0, sqrt(2.0f / input_size));
    return dist(Tools::gen);
}

/*
 * NEURON CLASS
 */
class Neuron {
public:
    explicit Neuron(int size) : weights_size(size), adam(size) {
        weight_grads = new float[weights_size];
        weights = new float[weights_size];
        bias_grad = 0;
        bias = 0;

        for (int i = 0; i < weights_size; ++i) {
            weight_grads[i] = 0;
            weights[i] = heInit(weights_size);
        }
    }

    ~Neuron() {
        delete[] weight_grads;
        delete[] weights;
    }

    float dotProduct(const float *input) const {
        __m512 sum = _mm512_setzero_ps();

        for (int i = 0; i < weights_size; i += 16) {
            __m512 va = _mm512_loadu_ps(input + i);
            __m512 vb = _mm512_loadu_ps(weights + i);
            sum = _mm512_fmadd_ps(va, vb, sum);
        }

        float result = _mm512_reduce_add_ps(sum);
        return result + bias;
    }

    void clearGradients() {
        __m512 zero = _mm512_setzero_ps();
        for (int i = 0; i < weights_size; i += 16) {
            _mm512_storeu_ps(weight_grads + i, zero);
        }
        bias_grad = 0;
    }

    void update(float lr) {
        adam.updateBias(lr, bias, bias_grad);
        adam.updateWeights(lr, weights, weight_grads);
    }

    void updateBiasGrad(float val) {
        bias_grad += val;
    }

    [[nodiscard]] int getSize() const {
        return weights_size;
    }

    [[nodiscard]] float *getWeights() const {
        return weights;
    }

    [[nodiscard]] float *getWeightGrads() const {
        return weight_grads;
    }

    void setWeights(float *newWeights) {
        delete[] weights;
        weights = newWeights;
    }

    [[nodiscard]] float getBias() const {
        return bias;
    }

    void setBias(float newBias) {
        bias = newBias;
    }

private:
    Adam adam;

    int weights_size;

    float *weight_grads;
    float bias_grad;
    float *weights;
    float bias;
};

#endif //ASTRA_NNETWORK_NEURON_H
