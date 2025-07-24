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
