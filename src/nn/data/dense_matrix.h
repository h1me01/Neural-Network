#pragma once

#include "array.h"

template <typename T> class DenseMatrix {
  public:
    DenseMatrix(int rows, int cols) //
        : rows(rows), cols(cols), data(rows * cols) {}

    DenseMatrix(const DenseMatrix &other) //
        : rows(other.rows), cols(other.cols), data(other.data) {}

    DenseMatrix(DenseMatrix &&other) noexcept //
        : rows(other.rows), cols(other.cols), data(std::move(other.data)) {}

    DenseMatrix &operator=(const DenseMatrix &other) {
        if(this != &other) {
            rows = other.rows;
            cols = other.cols;
            data = other.data;
        }
        return *this;
    }

    DenseMatrix &operator=(DenseMatrix &&other) noexcept {
        if(this != &other) {
            rows = other.rows;
            cols = other.cols;
            data = std::move(other.data);
        }
        return *this;
    }

    T &operator()(int index) {
        if(index >= data.size())
            error("DenseMatrix: index out of range");
        return data[index];
    }

    const T &operator()(int index) const {
        if(index >= data.size())
            error("DenseMatrix: index out of range");
        return data[index];
    }

    T &operator()(int row, int col) {
        if(row >= rows || col >= cols)
            error("DenseMatrix: index out of range");
        return data[row * cols + col];
    }

    const T &operator()(int row, int col) const {
        if(row >= rows || col >= cols)
            error("DenseMatrix: index out of range");
        return data[row * cols + col];
    }

    int num_rows() const {
        return rows;
    }

    int num_cols() const {
        return cols;
    }

    int size() const {
        return data.size();
    }

    void clear() {
        data.clear();
    }

  private:
    int rows;
    int cols;
    Array<T> data;
};
