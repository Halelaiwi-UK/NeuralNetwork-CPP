cmake_minimum_required(VERSION 3.29)
project(NeuralNetwork)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)

# Include project source directory for headers
include_directories(${PROJECT_SOURCE_DIR})

# Define the executable
add_executable(NeuralNetwork
        main.cpp
        NeuralNetwork.cpp
        NeuralNetwork.h
        UtilityFunctions.cpp
        UtilityFunctions.h
)
