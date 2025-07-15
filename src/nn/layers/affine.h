#pragma once

#include "layer.h"

template <int size> //
class Affine : public Layer {
  public:
    Affine(Layer *previous, WeightsInit weights_init) : previous(previous) {
        if(previous == nullptr)
            error("Affine layer must have a previous layer");

        weights = Tensor<float>(size, previous->output_size());

        weights.init(weights_init);
        biases.init(WeightsInit::Zero);
    }

    void forward() override {
        const auto &input_v = previous->get_output().get_values();
        auto &output_v = output.get_values();

        const auto &weights_v = weights.get_values();
        const auto &biases_v = biases.get_values();

        if(input_v.num_cols() != weights.num_cols())
            error("Input col size does not match weights col size");

        for(int i = 0; i < output_v.num_rows(); i++) {
            for(int j = 0; j < output_v.num_cols(); j++) {
                output_v(i, j) = biases_v(j);
                for(int k = 0; k < input_v.num_cols(); k++)
                    output_v(i, j) += weights_v(j, k) * input_v(i, k);
            }
        }
    }

    void backward() override {
        const auto &input_v = previous->get_output().get_values();
        const auto &output_g = output.get_gradients();

        const auto &weights_v = weights.get_values();
        auto &weights_g = weights.get_gradients();

        auto &biases_g = biases.get_gradients();

        for(int i = 0; i < weights.num_rows(); i++) {
            // weights gradient
            for(int j = 0; j < weights.num_cols(); j++)
                for(int k = 0; k < output_g.num_rows(); k++)
                    weights_g(i, j) += output_g(k, i) * input_v(k, j);
            // biases gradient
            for(int j = 0; j < output_g.num_rows(); j++)
                biases_g(i) += output_g(j, i);
        }

        // calculate gradients for previous layer
        auto &prev_g = previous->get_output().get_gradients();

        for(int i = 0; i < prev_g.num_rows(); i++) {
            for(int j = 0; j < prev_g.num_cols(); j++) {
                prev_g(i, j) = 0.0f;
                for(int k = 0; k < weights.num_rows(); k++)
                    prev_g(i, j) += output_g(i, k) * weights_v(k, j);
            }
        }
    }

    int input_size() const override {
        return previous->output_size();
    }

    int output_size() const override {
        return size;
    }

    std::vector<Tensor<float> *> get_params() override {
        return {&weights, &biases};
    }

    std::string get_info() const override {
        std::stringstream ss;
        ss << "Affine Layer: " << std::to_string(size) << " outputs, ";
        ss << std::to_string(input_size()) << " inputs";
        return ss.str();
    }

  private:
    Tensor<float> weights{1, 1};
    Tensor<float> biases{size, 1};

    Layer *previous;
};
