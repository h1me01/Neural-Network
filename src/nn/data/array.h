
#pragma once

#include <cstring>
#include <initializer_list>

#include "../../misc.h"

template <typename T> //
class Array {
  public:
    Array(int size) : m_size(size), data(new T[size]) {
        clear();
    }

    Array(std::initializer_list<T> init)                //
        : m_size(init.size()), data(new T[init.size()]) //
    {
        int i = 0;
        for(const auto &val : init)
            data[i++] = val;
    }

    Array(const Array &other)                             //
        : m_size(other.m_size), data(new T[other.m_size]) //
    {
        for(int i = 0; i < m_size; i++)
            data[i] = other.data[i];
    }

    Array(Array &&other) noexcept : m_size(other.m_size), data(other.data) {
        other.m_size = 0;
        other.data = nullptr;
    }

    Array &operator=(const Array &other) {
        if(this != &other) {
            delete[] data;
            m_size = other.m_size;
            data = new T[m_size];

            for(int i = 0; i < m_size; i++)
                data[i] = other.data[i];
        }

        return *this;
    }

    Array &operator=(Array &&other) noexcept {
        if(this != &other) {
            delete[] data;
            m_size = other.m_size;
            data = other.data;
            other.m_size = 0;
            other.data = nullptr;
        }

        return *this;
    }

    ~Array() {
        delete[] data;
    }

    void clear() {
        for(int i = 0; i < m_size; i++)
            data[i] = T();
    }

    T &operator[](int index) {
        return data[index];
    }

    const T &operator[](int index) const {
        return data[index];
    }

    int size() const {
        return m_size;
    }

  private:
    int m_size;
    T *data;
};
