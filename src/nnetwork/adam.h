#ifndef ASTRA_NNETWORK_ADAM_H
#define ASTRA_NNETWORK_ADAM_H

#include "../dataset.h"

class Adam {
public:
    explicit Adam(int numWeights) : weights_size(numWeights) {
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

    void updateWeights(const float lr, float *weights, const float *gradients) const {
        __m512 lr_avx = _mm512_set1_ps(lr);

        for (int i = 0; i < weights_size; i += 16) {
            __m512 gradient = _mm512_loadu_ps(gradients + i);
            __m512 m_weight = _mm512_loadu_ps(m_weights + i);
            __m512 v_weight = _mm512_loadu_ps(v_weights + i);
            __m512 weight = _mm512_loadu_ps(weights + i);

            m_weight = _mm512_add_ps(_mm512_mul_ps(beta1_avx, m_weight), _mm512_mul_ps(one_minus_beta1_avx, gradient));
            v_weight = _mm512_add_ps(_mm512_mul_ps(beta2_avx, v_weight), _mm512_mul_ps(one_minus_beta2_avx, _mm512_mul_ps(gradient, gradient)));

            __m512 denom = _mm512_add_ps(_mm512_sqrt_ps(v_weight), epsilon_avx);
            weight = _mm512_sub_ps(weight, _mm512_div_ps(_mm512_mul_ps(lr_avx, m_weight), denom));

            _mm512_storeu_ps(weights + i, weight);
            _mm512_storeu_ps(m_weights + i, m_weight);
            _mm512_storeu_ps(v_weights + i, v_weight);
        }
    }

    void updateBias(const float lr, float &bias, const float &gradient) {
        m_bias = beta1 * m_bias + (1 - beta1) * gradient;
        v_bias = beta2 * v_bias + (1 - beta2) * (gradient * gradient);
        bias -= lr * m_bias / (sqrt(v_bias) + epsilon);
    }

private:
    const float beta1 = 0.95f;
    const float beta2 = 0.999f;
    const float epsilon = 1e-8f;

    int weights_size;

    float *m_weights;
    float *v_weights;
    float m_bias = 0;
    float v_bias = 0;

    __m512 beta1_avx = _mm512_set1_ps(beta1);
    __m512 beta2_avx = _mm512_set1_ps(beta2);
    __m512 one_minus_beta1_avx = _mm512_set1_ps(1.0f - beta1);
    __m512 one_minus_beta2_avx = _mm512_set1_ps(1.0f - beta2);
    __m512 epsilon_avx = _mm512_set1_ps(epsilon);
};

#endif //ASTRA_NNETWORK_ADAM_H
