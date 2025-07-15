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

class MSE : public Loss {
  public:
    void compute(Tensor<float> &output, const std::vector<XorData> &data) override {
        const DenseMatrix<float> &predictions = output.get_values();
        DenseMatrix<float> &gradients = output.get_gradients();

        if(predictions.size() != data.size())
            error("Output and target sizes do not match");

        const int count = predictions.size();

        float sum = 0.0f;
        for(int i = 0; i < count; i++) {
            float diff = predictions(i) - data[i].get_target();
            sum += diff * diff;
            gradients(i) = 2.0f * diff;
        }

        loss = sum / count;
    }

    std::string get_info() const override {
        return "Mean Squared Error (MSE)";
    }
};

class MAE : public Loss {
  public:
    void compute(Tensor<float> &output, const std::vector<XorData> &data) override {
        const DenseMatrix<float> &predictions = output.get_values();
        DenseMatrix<float> &gradients = output.get_gradients();

        if(predictions.size() != data.size())
            error("Output and target sizes do not match");

        const int count = predictions.size();

        float sum = 0.0f;
        for(int i = 0; i < count; i++) {
            float diff = predictions(i) - data[i].get_target();
            sum += std::abs(diff);

            if(diff > 0.0f)
                gradients(i) = 1.0f;
            else if(diff < 0.0f)
                gradients(i) = -1.0f;
            else
                gradients(i) = 0.0f;
        }

        loss = sum / count;
    }

    std::string get_info() const override {
        return "Mean Absolute Error (MAE)";
    }
};
