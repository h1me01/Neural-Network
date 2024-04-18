#include "linkedList.h"

LinkedList::LinkedList() : head(nullptr) {}

LinkedList::~LinkedList() {
    while (head != nullptr) {
        Node *temp = head;
        head = head->next;
        delete temp;
    }
}

void LinkedList::insert(const SparseInput &data) {
    Node *newNode = new Node(data);
    newNode->next = head;
    head = newNode;
}

SparseInput& LinkedList::operator[](int index) {
    Node* current = head;

    int count = 0;
    while (current != nullptr && count < index) {
        current = current->next;
        count++;
    }

    if (current == nullptr) {
        throw std::out_of_range("Index out of range");
    }

    return current->data;
}

void LinkedList::shuffle() {
    if (head == nullptr || head->next == nullptr) {
        return; // no need to shuffle if list has 0 or 1 elements
    }

    uniform_int_distribution<int> dist;

    int count = 0;
    for (Node* current = head; current != nullptr; current = current->next) {
        count++;
    }

    // shuffle the list using Fisher-Yates algorithm
    for (int i = count - 1; i > 0; --i) {
        int j = dist(Tools::gen, decltype(dist)::param_type{0, i});

        Node* node_i = getNodeAtIndex(i);
        Node* node_j = getNodeAtIndex(j);

        swap(node_i->data, node_j->data);
    }
}

// helper to get the node at a given index
Node* LinkedList::getNodeAtIndex(int index) {
    Node* current = head;
    for (int i = 0; i < index && current != nullptr; ++i) {
        current = current->next;
    }
    return current;
}
