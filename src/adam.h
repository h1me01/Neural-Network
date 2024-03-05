#pragma once

#include "common.h"

class Adam {
public:
    explicit Adam(int numWeights) : numWeights(numWeights) {
        m_weights = new float[numWeights];
        v_weights = new float[numWeights];

        for (int i = 0; i < numWeights; ++i) {
            m_weights[i] = 0;
            v_weights[i] = 0;
        }

        m_bias = 0;
        v_bias = 0;
    }

    ~Adam() {
        delete[] m_weights;
        delete[] v_weights;
    }

    void updateWeights(float lr, float *weights, const float *gradients) {
        __m256 lr_avx = _mm256_set1_ps(lr);

        for (int i = 0; i < numWeights; i += 8) {
            __m256 gradient = _mm256_loadu_ps(gradients + i);
            __m256 m_weight = _mm256_loadu_ps(m_weights + i);
            __m256 v_weight = _mm256_loadu_ps(v_weights + i);
            __m256 weight = _mm256_loadu_ps(weights + i);

            m_weight = _mm256_add_ps(_mm256_mul_ps(beta1_avx, m_weight), _mm256_mul_ps(one_minus_beta1_avx, gradient));
            v_weight = _mm256_add_ps(_mm256_mul_ps(beta2_avx, v_weight), _mm256_mul_ps(one_minus_beta2_avx, _mm256_mul_ps(gradient, gradient)));

            __m256 denom = _mm256_add_ps(_mm256_sqrt_ps(v_weight), epsilon_avx);
            weight = _mm256_sub_ps(weight, _mm256_div_ps(_mm256_mul_ps(lr_avx, m_weight), denom));

            _mm256_storeu_ps(weights + i, weight);
            _mm256_storeu_ps(m_weights + i, m_weight);
            _mm256_storeu_ps(v_weights + i, v_weight);
        }
    }

    void updateBias(float lr, float &bias, const float &gradient) {
        m_bias = beta1 * m_bias + (1 - beta1) * gradient;
        v_bias = beta2 * v_bias + (1 - beta2) * (gradient * gradient);
        bias -= lr * m_bias / (sqrt(v_bias) + epsilon);
    }

private:
    int numWeights;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    float *m_weights;
    float *v_weights;
    float m_bias = 0;
    float v_bias = 0;

    __m256 beta1_avx = _mm256_set1_ps(beta1);
    __m256 beta2_avx = _mm256_set1_ps(beta2);
    __m256 one_minus_beta1_avx = _mm256_set1_ps(1.0f - beta1);
    __m256 one_minus_beta2_avx = _mm256_set1_ps(1.0f - beta2);
    __m256 epsilon_avx = _mm256_set1_ps(epsilon);

};
