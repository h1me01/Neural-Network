#ifndef ASTRA_NNETWORK_NEURON_H
#define ASTRA_NNETWORK_NEURON_H

#include "adam.h"

/*
 * HE INIT
 */
inline float heInit(int numPrevNeurons) {
    normal_distribution<float> dist(0, sqrt(2.0f / numPrevNeurons));
    return dist(Tools::gen);
}

/*
 * NEURON CLASS
 */
class Neuron {
public:
    explicit Neuron(int numFeatures) : numWeights(numFeatures), adam(numFeatures) {
        gradientWeights = new float[numWeights];
        weights = new float[numWeights];
        gradientBias = 0;
        bias = 0;

        for (int i = 0; i < numWeights; ++i) {
            gradientWeights[i] = 0;
            weights[i] = heInit(numWeights);
        }
    }

    ~Neuron() {
        delete[] gradientWeights;
        delete[] weights;
    }

    float dotProduct(const float *input) const {
        __m512 sum = _mm512_setzero_ps();

        for (int i = 0; i < numWeights; i += 16) {
            __m512 va = _mm512_loadu_ps(input + i);
            __m512 vb = _mm512_loadu_ps(weights + i);
            sum = _mm512_fmadd_ps(va, vb, sum);
        }

        float result = _mm512_reduce_add_ps(sum);
        return result + bias;
    }

    void clearGradients() {
        __m512 zero = _mm512_setzero_ps();
        for (int i = 0; i < numWeights; i += 16)
            _mm512_storeu_ps(gradientWeights + i, zero);

        gradientBias = 0;
    }

    void update(float lr) {
        adam.updateBias(lr, bias, gradientBias);
        adam.updateWeights(lr, weights, gradientWeights);
    }

    void updateGradientBias(float val) {
        gradientBias += val;
    }

    [[nodiscard]] int getNumWeights() const {
        return numWeights;
    }

    [[nodiscard]] float *getWeights() const {
        return weights;
    }

    [[nodiscard]] float *getGradientWeights() const {
        return gradientWeights;
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

    int numWeights;

    float *gradientWeights;
    float gradientBias;
    float *weights;
    float bias;
};

#endif //ASTRA_NNETWORK_NEURON_H
