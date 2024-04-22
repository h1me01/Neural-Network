#include "network.h"

Network::Network(bool loadWeights) {
    numLayers = 2;

    layers = new Layer *[numLayers];
    layers[0] = new Layer(INPUT_NEURONS, HIDDEN_NEURONS1, RELU);
    layers[1] = new Layer(HIDDEN_NEURONS1, OUTPUT_NEURONS, SIGMOID);

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

float Network::feedForward(SparseInput sparseInput) {
    float* input = sparseInput.input;

    for (int i = 0; i < numLayers; ++i) {
        input = layers[i]->feedForward(input);
    }

    return input[0];
}

void Network::feedBackward(float target) {
    // Update output layer
    layers[numLayers - 1]->calcOutputDelta(target);
    layers[numLayers - 1]->updateGradients();

    // Update hidden layer(s)
    for (int i = numLayers - 2; i >= 0; --i) {
        layers[i]->calcHiddenDeltas(layers[i + 1]);
        layers[i]->updateGradients();
    }
}

float Network::evaluate(string &fen) {
    float* input = fenToInput(fen).input;

    for (int i = 0; i < numLayers; ++i) {
        input = layers[i]->feedForward(input);
    }

    return 0 * 250 - 125;
}

void Network::save() {
    ofstream file(WEIGHTS_PATH, ios::binary);

    if (!file.is_open()) {
        cerr << "Error opening file for writing: " << WEIGHTS_PATH << endl;
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
    ifstream file(WEIGHTS_PATH, ios::binary);

    if (!file.is_open()) {
        cerr << "Error opening file for reading: " << WEIGHTS_PATH << endl;
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

void Network::train(vector<SparseInput> &data, const int epochs, const int batchSize) {
    shuffleData(data);

    int dataSize = data.size();
    int valSize = dataSize / 100;
    int trainingSize = dataSize - valSize;

    cout << "\nTraining Network with " << dataSize << " Positions\n" << endl;

    vector<SparseInput> valData(data.begin() + trainingSize, data.end());
    data.resize(trainingSize);

    auto startTime = chrono::high_resolution_clock::now();
    const int numBatches = (trainingSize + batchSize - 1) / batchSize;

    cout << left << setw(6) << "Epoch" << setw(4) << "|" << "Validation Loss" << endl;
    cout << "--------------------------------------" << endl;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        for (int batch = 0; batch < numBatches; ++batch) {
            int startIdx = batch * batchSize;
            int endIdx = min((batch + 1) * batchSize, trainingSize);

            for(int i = startIdx; i < endIdx; ++i) {
                feedForward(data[i]);
                feedBackward(data[i].target);
            }

            for (int i = 0; i < numLayers; ++i) {
                layers[i]->updateNeurons(LEARNING_RATE);
                layers[i]->clearAllGradients();
            }
        }

        if (epoch % 2 == 0) {
            float validationLoss = getLoss(valData);
            cout << setw(6) << epoch << setw(4) << "|" << validationLoss << endl;
        }
    }

    // save weights
    save();

    auto endTime = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(endTime - startTime);
    cout << "\nTraining Neural Network done! (" << duration.count() << " seconds)\n" << endl;
    cout << "--------------------------------------" << endl;
}
