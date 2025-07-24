#pragma once

#include "../data/include.h"

class Loss {
  public:
    Loss() {}
    ~Loss() = default;

    virtual void compute(Tensor<float> &output, const std::vector<XorData> &data) = 0;

    virtual std::string get_info() const = 0;

    float get_loss() const {
        return loss;
    }

  protected:
    float loss = 0;
};
