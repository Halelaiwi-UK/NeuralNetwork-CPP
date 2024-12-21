#include <iostream>
#include <NeuralNetwork.h>

int main() {
    auto network = NeuralNetwork(2);
    network.setLearningRate(0.001);
    network.add_layer(2);
    network.add_layer(1);
    auto input = std::vector<double>(2, 2);
    auto result = network.predict(input);
    std::cout << "[";
    for (auto& value : result) {
        std::cout << value << ", ";
    }
    std::cout << "]" << std::endl;
    network.train(input, std::vector<double>(1, 1), 10);
    result = network.predict(input);
    std::cout << "[";
    for (auto& value : result) {
        std::cout << value << ", ";
    }
    std::cout << "]" << std::endl;

    network.forwardPass(std::vector<double>(2, 2));
    return 0;
}
