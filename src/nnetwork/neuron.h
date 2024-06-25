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
    explicit Neuron(int numPrevNeurons) : numWeights(numPrevNeurons), adam(numPrevNeurons) {
        gradientWeights = new float[numWeights];
        weights = new float[numWeights];
        gradientBias = 0;
        bias = 0.01f;

        for (int i = 0; i < numWeights; ++i) {
            gradientWeights[i] = 0;
            weights[i] = heInit(numWeights);
        }
    }

    ~Neuron() {
        delete[] gradientWeights;
        delete[] weights;
    }

    float dotProduct(const float *input) {
        float dot = bias;

        __m256 num1, num2, num3, num4;
        num4 = _mm256_setzero_ps();

        for (int i = 0; i < numWeights; i += 8) {
            num1 = _mm256_loadu_ps(input + i);
            num2 = _mm256_loadu_ps(weights + i);
            num3 = _mm256_mul_ps(num1, num2);
            num4 = _mm256_add_ps(num4, num3);
        }

        num4 = _mm256_hadd_ps(num4, num4);
        num4 = _mm256_hadd_ps(num4, num4);

        __m128 lo = _mm256_castps256_ps128(num4);
        __m128 hi = _mm256_extractf128_ps(num4, 1);
        lo = _mm_add_ps(lo, hi);

        _mm_store_ss(&dot, lo);
        return dot;
    }

    void clearGradients() {
        __m256 zero = _mm256_setzero_ps();
        for (int i = 0; i < numWeights; i += 8)
            _mm256_storeu_ps(gradientWeights + i, zero);

        gradientBias = 0;
    }

    void update(float lr) {
        adam.updateBias(lr, bias, gradientBias);
        adam.updateWeights(lr, weights, gradientWeights);
    }

    void updateGradientWeight(int i, float val) const {
        gradientWeights[i] += val;
    }

    void updateGradientBias(float val) {
        gradientBias += val;
    }

    int getNumWeights() const { return numWeights; }
    float *getWeights() const { return weights; }

    void setWeights(float *newWeights) {
        delete[] weights;
        weights = newWeights;
    }

    float getBias() const { return bias; }
    void setBias(float newBias) { bias = newBias; }

private:
    Adam adam;

    int numWeights;

    float *gradientWeights;
    float gradientBias;
    float *weights;
    float bias;

};

#endif //ASTRA_NNETWORK_NEURON_H
