#pragma once

#include "optimizer.h"

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
