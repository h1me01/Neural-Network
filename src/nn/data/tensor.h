#pragma once

#include <random>

#include "../../misc.h"
#include "dense_matrix.h"

enum class WeightsInit { //
    Zero,
    He
};

template <typename T> //
class Tensor {
  public:
    Tensor(int rows, int cols) //
        : values(rows, cols), gradients(rows, cols) {}

    Tensor(const Tensor &other) //
        : values(other.values), gradients(other.gradients) {}

    Tensor(Tensor &&other) noexcept //
        : values(std::move(other.values)), gradients(std::move(other.gradients)) {}

    Tensor &operator=(const Tensor &other) {
        if(this != &other) {
            values = other.values;
            gradients = other.gradients;
        }
        return *this;
    }

    Tensor &operator=(Tensor &&other) noexcept {
        if(this != &other) {
            values = std::move(other.values);
            gradients = std::move(other.gradients);
        }
        return *this;
    }

    DenseMatrix<T> &get_values() {
        return values;
    }

    const DenseMatrix<T> &get_values() const {
        return values;
    }

    DenseMatrix<T> &get_gradients() {
        return gradients;
    }

    const DenseMatrix<T> &get_gradients() const {
        return gradients;
    }

    int num_rows() const {
        return values.num_rows();
    }

    int num_cols() const {
        return values.num_cols();
    }

    int size() const {
        return values.size();
    }

    void init(WeightsInit init) {
        gradients.clear();

        if(init == WeightsInit::Zero) {
            values.clear();
            return;
        }

        static std::mt19937 gen(std::random_device{}());
        std::normal_distribution<T> dist(0, sqrt(2.0 / values.num_cols()));

        for(int i = 0; i < values.size(); i++)
            values(i) = dist(gen);
    }

  private:
    DenseMatrix<T> values;
    DenseMatrix<T> gradients;
};
