#pragma once

#include "layer.h"

enum ActivationType { //
    Linear,
    ReLU,
    Sigmoid
};

template <ActivationType T> //
class Activation : public Layer {
  public:
    Activation(Layer *previous) : previous(previous) {
        if(previous == nullptr)
            error("Activation layer must have a previous layer");
    }

    void forward() override {
        const auto &input_v = previous->get_output().get_values();
        auto &output_v = output.get_values();

        for(int i = 0; i < input_v.size(); i++) {
            switch(T) {
            case Linear:
                output_v(i) = input_v(i);
                break;
            case Sigmoid:
                output_v(i) = 1.0f / (1.0f + std::exp(-input_v(i)));
                break;
            case ReLU:
                output_v(i) = std::max(0.0f, input_v(i));
                break;
            default:
                error("Unknown activation type");
                break;
            }
        }
    }

    void backward() override {
        const auto &input_v = previous->get_output().get_values();
        auto &input_g = previous->get_output().get_gradients();

        const auto &output_v = output.get_values();
        const auto &output_g = output.get_gradients();

        for(int i = 0; i < input_v.size(); i++) {
            float derivative = 0.0f;

            switch(T) {
            case Linear:
                derivative = 1.0f;
                break;
            case Sigmoid:
                derivative = output_v(i) * (1.0f - output_v(i));
                break;
            case ReLU:
                derivative = (input_v(i) > 0.0f) ? 1.0f : 0.0f;
                break;
            default:
                error("Unknown activation type");
                break;
            }

            input_g(i) = output_g(i) * derivative;
        }
    }

    int input_size() const override {
        return previous->output_size();
    }

    int output_size() const override {
        return previous->output_size();
    }

    std::vector<Tensor<float> *> get_params() override {
        return {};
    }

    std::string get_info() const override {
        return "Activation Layer (" + activation_string(T) + ") with size: " + std::to_string(output_size());
    }

  private:
    Layer *previous;
};