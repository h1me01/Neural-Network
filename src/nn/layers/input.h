#pragma once

#include "layer.h"

template <int size> //
struct Input : Layer {
    Input() {}

    void forward() override {}
    void backward() override {}

    int input_size() const override {
        return size;
    }

    int output_size() const override {
        return size;
    }

    std::vector<Tensor<float> *> get_params() override {
        return {};
    }

    std::string get_info() const override {
        return "Input Layer with size: " + std::to_string(size);
    }
};
