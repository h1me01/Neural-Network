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
        __m256 sum = _mm256_setzero_ps();

        for (int i = 0; i < numWeights; i += 8) {
            __m256 va = _mm256_loadu_ps(input + i);
            __m256 vb = _mm256_loadu_ps(weights + i);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
        }

        __m256 hsum = _mm256_hadd_ps(sum, sum);
        hsum = _mm256_hadd_ps(hsum, hsum);

        __m128 lo = _mm256_castps256_ps128(hsum);
        __m128 hi = _mm256_extractf128_ps(hsum, 1);
        __m128 result = _mm_add_ss(lo, hi);

        return _mm_cvtss_f32(result) + bias;
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
