#pragma once

#include "common.h"

class Adam {
public:
    explicit Adam(int numWeights);
    ~Adam();

    void updateWeights(float lr, float* weights, const float* gradients);
    void updateBias(float lr, float& bias, const float& gradient);

private:
    int numWeights;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    float* m_weights;
    float* v_weights;
    float m_bias = 0;
    float v_bias = 0;

    __m256 beta1_avx = _mm256_set1_ps(beta1);
    __m256 beta2_avx = _mm256_set1_ps(beta2);
    __m256 one_minus_beta1_avx = _mm256_set1_ps(1.0f - beta1);
    __m256 one_minus_beta2_avx = _mm256_set1_ps(1.0f - beta2);
    __m256 epsilon_avx = _mm256_set1_ps(epsilon);

};
