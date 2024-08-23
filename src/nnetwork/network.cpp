#include "network.h"

float Network::feedForward(NetInput &net_input) const {
    float *input = getSparseInput(net_input);
    for (int i = 0; i < num_layers; ++i) {
        input = layers[i]->feedForward(input);
    }
    return input[0];
}

void Network::feedBackward(float target) const {
    for (int i = num_layers - 1; i >= 0; --i) {
        if (i == num_layers - 1)
            layers[i]->calcDeltas(nullptr, target);
        else
            layers[i]->calcDeltas(layers[i + 1]);
        layers[i]->updateGradients();
    }
}

void Network::saveWeights(int epoch) const {
    const string weightsPath = "C:/Users/semio/Downloads/Astra-Weights/astra_weights_" + to_string(epoch) + ".nnue";

    ofstream file(weightsPath, ios::binary);

    if (!file.is_open()) {
        cerr << "Error opening file for writing: " << weightsPath << endl;
        return;
    }

    for (int i = 0; i < num_layers; ++i) {
        int input_size = layers[i]->getNumPrevNeurons();
        int output_size = layers[i]->getNumNeurons();

        file.write(reinterpret_cast<char *>(&input_size), sizeof(int));
        file.write(reinterpret_cast<char *>(&output_size), sizeof(int));

        for (int j = 0; j < output_size; ++j) {
            float *weights = layers[i]->getNeurons()[j]->getWeights();
            float bias = layers[i]->getNeurons()[j]->getBias();

            file.write(reinterpret_cast<char *>(weights), input_size * sizeof(float));
            file.write(reinterpret_cast<char *>(&bias), sizeof(float));
        }
    }

    file.close();
}

void Network::loadWeights(const string &weights_path) const {
    ifstream file(weights_path, ios::binary);

    if (!file.is_open()) {
        cerr << "Error opening file for reading: " << weights_path << endl;
        return;
    }

    for (int i = 0; i < num_layers; ++i) {
        int input_size;
        int output_size;

        file.read(reinterpret_cast<char *>(&input_size), sizeof(int));
        file.read(reinterpret_cast<char *>(&output_size), sizeof(int));

        for (int j = 0; j < output_size; ++j) {
            float *weights = new float[input_size];
            float bias;

            file.read(reinterpret_cast<char *>(weights), input_size * sizeof(float));
            file.read(reinterpret_cast<char *>(&bias), sizeof(float));

            layers[i]->getNeurons()[j]->setWeights(weights);
            layers[i]->getNeurons()[j]->setBias(bias);
        }
    }

    file.close();
}

void Network::train(vector<NetInput> &data, const int epochs, const int batch_size) {
    shuffleData(data);

    const int data_size = data.size();
    const int val_size = data_size / 100;
    const int training_size = data_size - val_size;
    const int num_batches = (training_size + batch_size - 1) / batch_size;

    cout << "\nTraining Network with " << data_size << " Positions\n" << endl;

    vector<NetInput> val_data(data.begin() + training_size, data.end());
    data.resize(training_size);

    auto start_time = chrono::high_resolution_clock::now();

    cout << left << setw(6) << "Epoch" << setw(4) << "|" << "Validation Loss" << endl;
    cout << "--------------------------------------" << endl;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        for (int batch = 0; batch < num_batches; ++batch) {
            int start_idx = batch * batch_size;
            int end_idx = min((batch + 1) * batch_size, training_size);

            for (int i = start_idx; i < end_idx; ++i) {
                feedForward(data[i]);
                feedBackward(data[i].target);
            }

            for (int i = 0; i < num_layers; ++i) {
                layers[i]->updateNeurons(LEARNING_RATE);
                layers[i]->clearAllGradients();
            }
        }

        if (epoch % 3 == 0) {
            float val_loss = getLoss(val_data);
            cout << setw(6) << epoch << setw(4) << "|" << val_loss << endl;
        }

        // save weights after every 5th epoch
        if (epoch % 5 == 0) {
            saveWeights(epoch);
        }
    }

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(end_time - start_time);
    cout << "\nTraining Neural Network done! (" << duration.count() << " seconds)\n" << endl;
    cout << "--------------------------------------" << endl;
}
