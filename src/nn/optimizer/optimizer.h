#pragma once

#include <cmath>

#include "../layers/include.h"

class Optimizer {
  public:
    Optimizer(float lr) : lr(lr) {};

    virtual ~Optimizer() = default;

    virtual void init(std::vector<Layer *> &layers) {
        for(auto layer : layers)
            for(auto *param : layer->get_params())
                params.push_back(param);
    }

    virtual void update(int batch_size) = 0;

    virtual std::string get_info() const = 0;

  protected:
    float lr;
    std::vector<Tensor<float> *> params;
};

class SGD : public Optimizer {
  public:
    SGD(float lr) : Optimizer(lr) {}

    void update(int batch_size) override {
        const float grad_scale = 1.0f / batch_size;

        for(auto *param : params) {
            auto &values = param->get_values();
            auto &gradients = param->get_gradients();

            for(int i = 0; i < values.size(); i++) {
                values(i) -= lr * gradients(i) * grad_scale;
                gradients(i) = 0;
            }
        }
    }

    std::string get_info() const override {
        return "Stochastic Gradient Descent (SGD)";
    }
};

class Adam : public Optimizer {
  public:
    Adam(float lr,             //
         float beta1 = 0.9f,   //
         float beta2 = 0.999f, //
         float eps = 1e-8f,    //
         float decay = 0.01f)
        : Optimizer(lr), beta1(beta1), beta2(beta2), eps(eps), decay(decay) {}

    virtual void init(std::vector<Layer *> &layers) override {
        Optimizer::init(layers);

        for(int i = 0; i < params.size(); i++) {
            momentums.emplace_back(Array<float>(params[i]->size()));
            velocities.emplace_back(Array<float>(params[i]->size()));
        }
    }

    void update(int batch_size) override {
        const float grad_scale = 1.0f / batch_size;
        const float _decay = 1 - lr * decay;

        for(int i = 0; i < params.size(); i++) {
            auto &values = params[i]->get_values();
            auto &gradients = params[i]->get_gradients();
            auto &m = this->momentums[i];
            auto &v = this->velocities[i];

            for(int j = 0; j < values.size(); j++) {
                const float grad = gradients(j) * grad_scale;

                values(j) *= _decay;

                m[j] = beta1 * m[j] + (1 - beta1) * grad;
                v[j] = beta2 * v[j] + (1 - beta2) * grad * grad;

                values(j) -= lr * m[j] / (sqrt(v[j]) + eps);
                gradients(j) = 0;
            }
        }
    }

    std::string get_info() const override {
        return "Adam Optimizer";
    }

  private:
    float beta1;
    float beta2;
    float eps;
    float decay;

    std::vector<Array<float>> momentums;
    std::vector<Array<float>> velocities;
};
