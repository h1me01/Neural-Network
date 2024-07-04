#include "network.h"

float Network::feedForward(NetInput& netInput) const {
    float* input = getSparseInput(netInput);
    for (int i = 0; i < numLayers; ++i)
        input = layers[i]->feedForward(input);
    return input[0];
}

void Network::feedBackward(float target) const {
    // Update output layer
    layers[numLayers - 1]->calcOutputDelta(target);
    layers[numLayers - 1]->updateGradients();

    // Update hidden layer(s)
    for (int i = numLayers - 2; i >= 0; --i) {
        layers[i]->calcHiddenDeltas(layers[i + 1]);
        layers[i]->updateGradients();
    }
}

void Network::saveWeights(int epoch) const {
    const string weightsPath = "C:/Users/semio/Downloads/Astra-Weights/astra_weights_" + to_string(epoch) + "_768-64-1.nnue";

    ofstream file(weightsPath, ios::binary);

    if (!file.is_open()) {
        cerr << "Error opening file for writing: " << weightsPath << endl;
        return;
    }

    for (int i = 0; i < numLayers; ++i) {
        int numPrevNeurons = layers[i]->getNumPrevNeurons();
        int numNeurons = layers[i]->getNumNeurons();

        file.write(reinterpret_cast<char*>(&numPrevNeurons), sizeof(int));
        file.write(reinterpret_cast<char*>(&numNeurons), sizeof(int));

        for (int j = 0; j < numNeurons; ++j) {
            float* weights = layers[i]->getNeurons()[j]->getWeights();
            float bias = layers[i]->getNeurons()[j]->getBias();

            file.write(reinterpret_cast<char*>(weights), numPrevNeurons * sizeof(float));
            file.write(reinterpret_cast<char*>(&bias), sizeof(float));
        }
    }

    file.close();
}

void Network::loadWeights(const string& weightsPath) const {
    ifstream file(weightsPath, ios::binary);

    if (!file.is_open()) {
        cerr << "Error opening file for reading: " << weightsPath << endl;
        return;
    }

    for (int i = 0; i < numLayers; ++i) {
        int numPrevNeurons;
        int numNeurons;

        file.read(reinterpret_cast<char*>(&numPrevNeurons), sizeof(int));
        file.read(reinterpret_cast<char*>(&numNeurons), sizeof(int));

        for (int j = 0; j < numNeurons; ++j) {
            float* weights = new float[numPrevNeurons];
            float bias;

            file.read(reinterpret_cast<char*>(weights), numPrevNeurons * sizeof(float));
            file.read(reinterpret_cast<char*>(&bias), sizeof(float));

            layers[i]->getNeurons()[j]->setWeights(weights);
            layers[i]->getNeurons()[j]->setBias(bias);
        }
    }

    file.close();
}

void Network::train(vector<NetInput>& data, const int epochs, const int batchSize) {
    shuffleData(data);

    const int dataSize = data.size();
    const int valSize = dataSize / 100;
    const int trainingSize = dataSize - valSize;
    const int numBatches = (trainingSize + batchSize - 1) / batchSize;

    cout << "\nTraining Network with " << dataSize << " Positions\n" << endl;

    vector<NetInput> valData(data.begin() + trainingSize, data.end());
    data.resize(trainingSize);

    auto startTime = chrono::high_resolution_clock::now();

    cout << left << setw(6) << "Epoch" << setw(4) << "|" << "Validation Loss" << endl;
    cout << "--------------------------------------" << endl;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        for (int batch = 0; batch < numBatches; ++batch) {
            int startIdx = batch * batchSize;
            int endIdx = min((batch + 1) * batchSize, trainingSize);

            for (int i = startIdx; i < endIdx; ++i) {
                feedForward(data[i]);
                feedBackward(data[i].target);
            }

            for (int i = 0; i < numLayers; ++i) {
                layers[i]->updateNeurons(LEARNING_RATE);
                layers[i]->clearAllGradients();
            }
        }

        if(epoch % 2 == 0) {
            float validationLoss = getLoss(valData);
            cout << setw(6) << epoch << setw(4) << "|" << validationLoss << endl;
        }

        // save weights after each epoch
        //saveWeights(epoch);
    }

    auto endTime = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(endTime - startTime);
    cout << "\nTraining Neural Network done! (" << duration.count() << " seconds)\n" << endl;
    cout << "--------------------------------------" << endl;
}
