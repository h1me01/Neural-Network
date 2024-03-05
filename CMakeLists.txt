cmake_minimum_required(VERSION 3.27)
project(Astra_NNetwork)

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

add_executable(Astra_NNetwork
        src/main.cpp
        src/common.h
        src/network.h
        src/neuron.h
        src/adam.h
        src/layer.h
        src/layer.cpp
        src/neuron.cpp
        src/network.cpp
)