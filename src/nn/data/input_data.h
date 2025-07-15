#pragma once

#include "array.h"

class XorData {
  public:
    XorData(Array<int> data, float target) : data(data), target(target) {}

    XorData(const XorData &other) //
        : data(other.data), target(other.target) {}

    XorData(XorData &&other) noexcept //
        : data(std::move(other.data)), target(other.target) {}

    XorData &operator=(const XorData &other) {
        if(this != &other) {
            data = other.data;
            target = other.target;
        }
        return *this;
    }

    XorData &operator=(XorData &&other) noexcept {
        if(this != &other) {
            data = std::move(other.data);
            target = other.target;
        }
        return *this;
    }

    Array<int> &get_data() {
        return data;
    }

    const Array<int> &get_data() const {
        return data;
    }

    float get_target() const {
        return target;
    }

  private:
    Array<int> data;
    float target;
};
