#pragma once

#include <iostream>
#include <sstream>
#include <string>

void error(std::string message) {
    std::cerr << "Error: " << message << std::endl;
    exit(EXIT_FAILURE);
}

std::string activation_string(int act_type) {
    switch(act_type) {
        // clang-format off
        case 0: return "Linear";
        case 1: return "Sigmoid";
        case 2: return "ReLU";
        default: return "Unknown Activation Type";
        // clang-format on
    }
}