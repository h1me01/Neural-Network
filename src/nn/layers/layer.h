#pragma once

#include <vector>

#include "../data/include.h"

class Layer {
  public:
    Layer() = default;

    virtual ~Layer() = default;

    void init(int batch_size) {
        output = Tensor<float>(batch_size, output_size());
    }

    virtual void forward() = 0;
    virtual void backward() = 0;

    virtual int input_size() const = 0;
    virtual int output_size() const = 0;

    Tensor<float> &get_output() {
        return output;
    }

    virtual std::vector<Tensor<float> *> get_params() = 0;
    virtual std::string get_info() const = 0;

  protected:
    Tensor<float> output{1, 1};
};
