#ifndef ASTRA_NNETWORK_LINKEDLIST_H
#define ASTRA_NNETWORK_LINKEDLIST_H

#include "dataset.h"

struct Node {
    SparseInput data;
    Node *next;

    Node(const SparseInput &d) : data(d), next(nullptr) {}
};

class LinkedList {
public:
    int size;

    LinkedList();

    ~LinkedList();

    void insert(const SparseInput &data);

    SparseInput& operator[](int index);

    void shuffle();

    // helper to get the node at a given index
    Node* getNodeAtIndex(int index);
private:
    Node *head;
};


#endif //ASTRA_NNETWORK_LINKEDLIST_H
