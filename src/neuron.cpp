#include "neuron.h"

/*
 * HE INIT
 */
float heInit(int numPrevNeurons) {
    normal_distribution<float> dist(0, sqrt(2.0f / numPrevNeurons));
    return dist(Tools::gen);
}

/*
 * NEURON CLASS
 */
Neuron::Neuron(int numPrevNeurons) : numWeights(numPrevNeurons), adam(numPrevNeurons) {
    gradientWeights = new float[numWeights];
    weights = new float[numWeights];
    gradientBias = 0;
    bias = 0.01f;

    for (int i = 0; i < numWeights; ++i) {
        gradientWeights[i] = 0;
        weights[i] = heInit(numWeights);
    }
}

Neuron::~Neuron() {
    delete[] gradientWeights;
    delete[] weights;
}

float Neuron::dotProduct(const float *input) {
    float total;

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

    _mm_store_ss(&total, lo);
    return total + bias;
}

void Neuron::clearGradients() {
    __m256 zero = _mm256_setzero_ps();
    for (int i = 0; i < numWeights; i += 8) {
        _mm256_storeu_ps(gradientWeights + i, zero);
    }

    gradientBias = 0;
}

void Neuron::update(float lr) {
    adam.updateBias(lr, bias, gradientBias);
    adam.updateWeights(lr, weights, gradientWeights);
}
