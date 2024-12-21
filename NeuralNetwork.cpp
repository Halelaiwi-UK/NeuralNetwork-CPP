#include "NeuralNetwork.h"
#include "UtilityFunctions.h"

#include <iostream>
#include <stdexcept>
#include <execution>
#include <omp.h>

NeuralNetwork::NeuralNetwork(const int input_size): last_layer_size(input_size), total_error(0),
                                                    gen(std::random_device{}()) {
    if (input_size <= 0) {
        throw std::invalid_argument("Input size must be positive.");
    }
}

void NeuralNetwork::setLearningRate(const double value) {
    this->learning_rate = value > 0 ? value : 0.01;
}

void NeuralNetwork::add_layer(const int layer_size) {
    if (layer_size <= 0) {
        throw std::invalid_argument("Layer size must be bigger than 0");
    }

    const int cols = this->last_layer_size;
    const int rows = layer_size;
    weightsMatrices.emplace_back(rows, std::vector<double>(cols));
    biasVectors.emplace_back(rows, 0.0);

    auto& layerMatrix = weightsMatrices.back();
    auto& layerBiases = biasVectors.back();
    std::normal_distribution<> distrib(0.0, std::sqrt(1.0 / cols)); // He Initialization

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            layerMatrix[i][j] = distrib(gen);
        }
        layerBiases[i] = 0.0; // Initialize biases similarly
    }

    last_layer_size = layer_size;
}


void NeuralNetwork::printStructure() {
    if (this->weightsMatrices.empty()) {
        std::cout << "Empty network" << std::endl;
        return;
    }
    int layerNum = 1;
    for (auto& layer : weightsMatrices) {
        std::cout << "Layer #" << layerNum << ": " << std::endl;
        for (auto nodeNum = 1 ; nodeNum <= layer.size() ; nodeNum++) {
            std::cout << "Node #" << nodeNum << "[";
            for (const auto& val : layer[nodeNum-1]) {
                std::cout << val << ",";
            }
            std::cout << "]" << std::endl;
            std::cout << " Bias : " << biasVectors[layerNum-1][nodeNum-1]<< std::endl;
        }
        layerNum++;
        std::cout << std::endl;
    }
}

void NeuralNetwork::backPropagate(const std::vector<double>& actual, const std::vector<double>& expected, double learning_rate) {
    if (actual.size() != expected.size()) {
        throw std::invalid_argument("Actual size does not match expected size.");
    }
    if (weightsMatrices.empty()) {
        throw std::invalid_argument("Empty network");
    }

    // Compute error for the output layer
    std::vector<double> outputError(actual.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        outputError[i] = actual[i] - expected[i];
    }

    // Gradients for weights and biases
    std::vector<std::vector<std::vector<double>>> weightGradients(weightsMatrices.size());
    std::vector<std::vector<double>> biasGradients(biasVectors.size());

    // Backpropagate through layers
    std::vector<double> prevLayerError = outputError;

    for (int layer = weightsMatrices.size() - 1; layer >= 0; --layer) {
        const auto& weightMatrix = weightsMatrices[layer];
        std::vector<double> currentLayerError(weightMatrix[0].size(), 0.0);
        std::vector<double> layerOutput = layer == 0 ? input : layerOutputs[layer - 1];

        // Gradients for weights and biases
        weightGradients[layer] = std::vector<std::vector<double>>(weightMatrix.size(), std::vector<double>(weightMatrix[0].size(), 0.0));
        biasGradients[layer] = std::vector<double>(biasVectors[layer].size(), 0.0);
        #pragma omp parallel for
        for (std::size_t neuron = 0; neuron < weightMatrix.size(); ++neuron) {
            double gradient_clip_threshold = 5.0;
            double delta = (layer == weightsMatrices.size() - 1)
                               ? outputError[neuron]
                               : prevLayerError[neuron] * UtilityFunctions::SigmoidDerivative(layerOutput[neuron]);

            // Clip gradients
            delta = std::max(std::min(delta, gradient_clip_threshold), -gradient_clip_threshold);
            #pragma omp parallel for
            for (std::size_t weight = 0; weight < weightMatrix[neuron].size(); ++weight) {
                weightGradients[layer][neuron][weight] = delta * layerOutput[weight];
                currentLayerError[weight] += delta * weightMatrix[neuron][weight];
            }
            biasGradients[layer][neuron] = delta;
        }

        prevLayerError = currentLayerError; // Update error for the next layer
    }

    // Update weights and biases
    for (std::size_t layer = 0; layer < weightsMatrices.size(); ++layer) {
        for (std::size_t neuron = 0; neuron < weightsMatrices[layer].size(); ++neuron) {
            for (std::size_t weight = 0; weight < weightsMatrices[layer][neuron].size(); ++weight) {
                weightsMatrices[layer][neuron][weight] -= learning_rate * weightGradients[layer][neuron][weight];
            }
            biasVectors[layer][neuron] -= learning_rate * biasGradients[layer][neuron];
        }
    }
}
void NeuralNetwork::train(std::vector<std::vector<double>>& input, std::vector<std::vector<double>>& expected, int epochs) {
    total_error = 0;
#   pragma omp parallel for
    for (std::size_t epoch = 0; epoch < epochs; epoch++) {
        std::cout << "Epoch: " << epoch << std::endl;
        std::vector<double> EpochError;
        #pragma omp parallel for
        for (int i = 0; i < input.size() / 100; i++){
            this->forwardPass(input[i]);
            this->backPropagate(this->output, expected[i], 0.1);
            auto error = UtilityFunctions::MSE(this->output, expected[i]);
            EpochError.push_back(std::reduce(error.begin(), error.end(), 0.0));
        }
        double epochTotalError = std::reduce(EpochError.begin(), EpochError.end(), 0.0);
        std::cout << "Epoch #" << epoch << " error: " << epochTotalError << std::endl;
    }

    #pragma omp parallel for reduction(+:total_error)
    for (std::size_t i = 0; i < expected.size(); i++) {
        auto error = UtilityFunctions::MSE(this->output, expected[i]);
        double partial_error = std::reduce(std::execution::seq, error.begin(), error.end(), 0.0);
        total_error += partial_error;
    }
    std::cout << "Final total error: " << total_error << std::endl;
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& input) {
    this->forwardPass(input);
    return this->output;
}

void NeuralNetwork::addWeightLayer(const std::vector<std::vector<double>>& weights) {
    // construct element in place and push it to weights vector
    auto& layer = weightsMatrices.back();
    auto& layerBiases = biasVectors.back();
    layer = weights;
    layerBiases = std::vector<double>(last_layer_size, 0.0);
}

void NeuralNetwork::forwardPass(const std::vector<double>& input) {
    this->input = input;
    this->layerOutputs.clear();
    auto prev = input;
    for (int i = 0; i < weightsMatrices.size(); ++i) {
        prev = UtilityFunctions::multiplyMatrixVector(weightsMatrices[i], prev);
        prev = UtilityFunctions::VectorAddition(prev, biasVectors[i]);
        if (i == weightsMatrices.size() - 1) {
            prev = UtilityFunctions::Softmax(prev); // Apply Softmax for output layer
        } else {
            prev = UtilityFunctions::SigmoidVector(prev);
        }
        layerOutputs.push_back(prev);
    }
    this->output = prev;
}