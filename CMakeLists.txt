cmake_minimum_required(VERSION 3.27)
project(Astra_NNetwork)

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

add_executable(Astra_NNetwork
        src/main.cpp
        src/misc.h
        src/misc.cpp
        src/dataset.h
        src/dataset.cpp
        src/nnetwork/adam.h
        src/nnetwork/adam.cpp
        src/nnetwork/neuron.h
        src/nnetwork/neuron.cpp
        src/nnetwork/layer.h
        src/nnetwork/layer.cpp
        src/nnetwork/network.h
        src/nnetwork/network.cpp
)