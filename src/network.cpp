#include "network.h"

Network::Network(bool loadWeights) {
    numLayers = 3;
    layers = new Layer *[numLayers];

    layers[0] = new Layer(INPUT_NEURONS, HIDDEN_NEURONS1, RELU);
    layers[1] = new Layer(HIDDEN_NEURONS1, HIDDEN_NEURONS2, RELU);
    layers[2] = new Layer(HIDDEN_NEURONS2, OUTPUT_NEURONS, SIGMOID);

    if (loadWeights) {
        load();
    }
}

Network::~Network() {
    for (int i = 0; i < numLayers; ++i) {
        delete layers[i];
    }
    delete[] layers;
}

float Network::feedForward(const NetInput &netInput) {
    float input[2 * 6 * 64];

    int inputIndex = 0;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 64; ++j) {
            input[inputIndex++] = getBit(netInput.pieces[WHITE][i], j);
        }

        for (int j = 0; j < 64; ++j) {
            input[inputIndex++] = getBit(netInput.pieces[BLACK][i], j);
        }
    }

    for (int i = 0; i < numLayers; ++i) {
        float *layerOutput = layers[i]->feedForward(input);
        copy_n(layerOutput, layers[i]->getNumNeurons(), input);
    }

    return input[0];
}

void Network::feedBackward(float target) {
    // Update output layer
    float outputDelta = layers[numLayers - 1]->calcOutputDelta(target);
    vector<float> deltas = {outputDelta};
    layers[numLayers - 1]->updateGradients(deltas.data());

    // Update hidden layer(s)
    for (int i = numLayers - 2; i >= 0; --i) {
        float *currentDeltas = layers[i]->calcHiddenDeltas(layers[i + 1], deltas.data());
        deltas.resize(layers[i]->getNumNeurons());
        copy_n(currentDeltas, layers[i]->getNumNeurons(), deltas.begin());
        layers[i]->updateGradients(deltas.data());
    }
}

float Network::evaluate(string &fen) {
    vector<float> input = fenToInput(fen);

    for (int i = 0; i < numLayers; ++i) {
        float *layerOutput = layers[i]->feedForward(input.data());
        input.resize(layers[i]->getNumNeurons());
        copy_n(layerOutput, layers[i]->getNumNeurons(), input.begin());
    }

    return input[0] * 250 - 125;
}

void Network::save() {
    std::ofstream file(WEIGHTS_PATH, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error opening file for writing: " << WEIGHTS_PATH << std::endl;
        return;
    }

    for (int i = 0; i < numLayers; ++i) {
        int numPrevNeurons = layers[i]->getNumPrevNeurons();
        int numNeurons = layers[i]->getNumNeurons();

        file.write(reinterpret_cast<char *>(&numPrevNeurons), sizeof(int));
        file.write(reinterpret_cast<char *>(&numNeurons), sizeof(int));

        for (int j = 0; j < numNeurons; ++j) {
            float *weights = layers[i]->getNeurons()[j]->getWeights();
            float bias = layers[i]->getNeurons()[j]->getBias();

            file.write(reinterpret_cast<char *>(weights), numPrevNeurons * sizeof(float));
            file.write(reinterpret_cast<char *>(&bias), sizeof(float));
        }
    }

    file.close();
}

void Network::load() {
    std::ifstream file(WEIGHTS_PATH, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error opening file for reading: " << WEIGHTS_PATH << std::endl;
        return;
    }

    for (int i = 0; i < numLayers; ++i) {
        int numPrevNeurons;
        int numNeurons;

        file.read(reinterpret_cast<char *>(&numPrevNeurons), sizeof(int));
        file.read(reinterpret_cast<char *>(&numNeurons), sizeof(int));

        for (int j = 0; j < numNeurons; ++j) {
            float *weights = new float[numPrevNeurons];
            float bias;

            file.read(reinterpret_cast<char *>(weights), numPrevNeurons * sizeof(float));
            file.read(reinterpret_cast<char *>(&bias), sizeof(float));

            layers[i]->getNeurons()[j]->setWeights(weights);
            layers[i]->getNeurons()[j]->setBias(bias);
        }
    }

    file.close();
}

void Network::train(vector<NetInput> &data, const int epochs, const int batchSize) {
    int dataSize = data.size();
    int valSize = dataSize / 100;
    int trainingSize = dataSize - valSize;

    cout << "\nTraining Network with " << dataSize << " Positions\n" << endl;

    vector<NetInput> valData(data.begin() + trainingSize, data.end());
    data.resize(trainingSize);

    auto startTime = chrono::high_resolution_clock::now();
    const int numBatches = (trainingSize + batchSize - 1) / batchSize;

    cout << left << setw(6) << "Epoch" << setw(4) << "|" << "Validation Cost" << endl;
    cout << "--------------------------------------" << endl;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        shuffleData(data);

        for (int batch = 0; batch < numBatches; ++batch) {
            int startIdx = batch * batchSize;
            int endIdx = min((batch + 1) * batchSize, trainingSize);
            vector<NetInput> minibatch(data.begin() + startIdx, data.begin() + endIdx);

            for (auto &d: minibatch) {
                feedForward(d);
                feedBackward(d.target);
            }

            for (int i = 0; i < numLayers; ++i) {
                layers[i]->updateNeurons(LEARNING_RATE);
                layers[i]->clearAllGradients();
            }
        }

        if (epoch % 2 == 0) {
            float validationCost = cost(valData);
            cout << setw(6) << epoch << setw(4) << "|" << validationCost << endl;
        }
    }

    save();

    auto endTime = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(endTime - startTime);
    cout << "\nTraining Neural Network done! (" << duration.count() << " seconds)\n" << endl;
    cout << "--------------------------------------" << endl;
}
