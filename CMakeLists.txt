# Minimum required version of CMake
cmake_minimum_required(VERSION 3.28)
project(Neural-Net)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -funroll-loops -march=native")

add_executable(Neural-Network src/main.cpp)
