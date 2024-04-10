#ifndef ASTRA_NNETWORK_DATASET_H
#define ASTRA_NNETWORK_DATASET_H

#include "misc.h"

void shuffleData(vector<SparseInput> &data);

int pieceIndex(char c);

int mirrorVertically(int sq);

int index(int psq, char p, Color view);
int index(int psq, int pt, Color pc, Color view);

SparseInput getSparseInput(NetInput &netInput);

vector<SparseInput> getSparseData(const string &filePath, int dataSize = INT_MAX);

SparseInput fenToInput(string &fen);

vector<float> normalizeTargets(vector<float> &targetValues);

#endif //ASTRA_NNETWORK_DATASET_H
