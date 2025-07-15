#pragma once

#include <algorithm>
#include <iomanip>

#include "data/include.h"
#include "layers/include.h"
#include "loss/loss.h"
#include "optimizer/optimizer.h"

class Network {
  public:
    explicit Network(               //
        int epochs,                 //
        int batch_size,             //
        Loss *loss,                 //
        Optimizer *optim,           //
        std::vector<Layer *> layers //
    ) {
        this->epochs = epochs;
        this->batch_size = batch_size;
        this->loss = loss;
        this->optim = optim;
        this->layers = std::move(layers);
    }

    ~Network() {
        for(auto layer : layers)
            delete layer;
        delete optim;
        delete loss;
    }

    float predict(const XorData &data) {
        batch_size = 1;
        init();
        forward({data});
        return layers.back()->get_output().get_values()(0);
    }

    void train(std::vector<XorData> &data) {
        init();
        print_info();
        std::cout << "================   Training   ================\n\n";

        batch_size = std::min(batch_size, (int) data.size());
        const int num_batches = (data.size() + batch_size - 1) / batch_size;

        for(int e = 1; e <= epochs; e++) {
            shuffle_data(data);
            float epoch_loss = 0.0f;

            for(int b = 0; b < num_batches; b++) {
                std::vector<XorData> batch_data;
                batch_data.reserve(batch_size);

                const int b_start = b * batch_size;

                // fill batch_data with exactly batch_size elements
                for(int i = 0; i < batch_size; i++) {
                    int data_idx = (b_start + i) % data.size();
                    batch_data.push_back(data[data_idx]);
                }

                forward(batch_data);
                backward(batch_data);
                optim->update(batch_size);
                epoch_loss += loss->get_loss();
            }

            if(e % std::max(epochs / 10, 1) == 0 || e == epochs) {
                std::cout << " epoch = " << std::setw(std::to_string(epochs).length()) << e
                          << ", loss = " << epoch_loss / num_batches << std::endl;
            }
        }
    }

  private:
    int epochs;
    int batch_size;

    std::vector<Layer *> layers;
    Optimizer *optim;
    Loss *loss;

    void init() {
        for(auto &layer : layers)
            layer->init(batch_size);
        optim->init(layers);
    }

    void print_info() const {
        std::stringstream ss;

        ss << "\n================ Network Info ================\n\n";
        ss << " Epochs:     " << epochs << std::endl;
        ss << " Batch Size: " << batch_size << std::endl;
        ss << " Loss:       " << loss->get_info() << std::endl;
        ss << " Optimizer:  " << optim->get_info() << std::endl;

        ss << "\n================ Network Arch ================\n\n";
        for(const auto &l : layers)
            ss << " -> " << l->get_info() << std::endl;

        std::cout << ss.str() << std::endl;
    }

    void forward(const std::vector<XorData> &data) {
        fill(data);
        for(auto &layer : layers)
            layer->forward();
    }

    void backward(const std::vector<XorData> &data) {
        loss->compute(layers.back()->get_output(), data);
        for(int i = int(layers.size()) - 1; i >= 0; i--)
            layers[i]->backward();
    }

    void fill(const std::vector<XorData> &data) {
        auto &input_v = layers[0]->get_output().get_values();
        if(input_v.size() != data.size() * data[0].get_data().size())
            error("Input size does not match data size");

        for(int i = 0; i < data.size(); i++) {
            auto &d = data[i].get_data();
            for(int j = 0; j < d.size(); j++)
                input_v(i, j) = d[j];
        }
    }

    void shuffle_data(std::vector<XorData> &data) {
        static std::random_device rd;
        static std::mt19937 g(rd());
        std::shuffle(data.begin(), data.end(), g);
    }
};
