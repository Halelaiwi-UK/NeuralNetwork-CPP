#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <random>

class NeuralNetwork {
private:
    std::vector<std::vector<std::vector<double>>> weightsMatrices;
    std::vector<std::vector<double>> biasVectors;
    std::vector<std::vector<double>> layerOutputs;
    std::vector<double> output;
    std::vector<double> input;
    std::mt19937 gen;
    double learning_rate = 0.01;
    int last_layer_size;
    double total_error;

public:
    explicit NeuralNetwork(int input_size);

    void setLearningRate(double value);
    void addWeightLayer(const std::vector<std::vector<double>>& weights);

    void forwardPass(const std::vector<double>& input);

    void add_layer(int layer_size);

    void printStructure();

    void backPropagate(const std::vector<double>& actual, const std::vector<double>& expected, double learning_rate);

    void train(std::vector<std::vector<double>>& input, std::vector<std::vector<double>>& expected, int epochs);

    std::vector<double> predict(const std::vector<double> &input);
};

#endif // NEURALNETWORK_H
