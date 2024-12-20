#include <iostream>
#include <random>
#include <list>

std::vector<double> multiplyMatrixVector(const std::vector<std::vector<double>>& matrix, const std::vector<double>& vec) {
    int rows = matrix.size();
    int cols = vec.size();
    std::vector<double> result(rows, 0.0);

    // Check for valid dimensions
    for (const auto& row : matrix) {
        if (row.size() != cols) {
            throw std::invalid_argument("Matrix columns must match vector size");
        }
    }

    // Matrix-vector multiplication
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i] += matrix[i][j] * vec[j];
        }
    }
    return result;
}

class Node {
    private:
        double *weights;
        double bias;
    public:
        Node(int n_weights) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> distrib(-1, 1);

            weights = new double[n_weights];
            for (int i = 0; i < n_weights; i++) {
                weights[i] = distrib(gen);
            }
            bias = distrib(gen);
        }
};

class NeuralNetwork {
private:
    std::list<std::vector<std::vector<double>>> weightsMatrices;
    std::list<std::vector<double>> biasVectors;
    int last_layer_size;
public:
    NeuralNetwork(int input_size) {
        // for adding layers, as of now, empty network
        this->last_layer_size = input_size;
    }

    void add_layer(int size) {
        // initialize vector
        int cols = this->last_layer_size;
        int rows = size;
        std::vector layerMatrix(rows, std::vector<double>(cols));
        std::vector<double> layerBiases(rows);

        std::random_device rd;
        std::mt19937 gen(rd()); // Random number generator
        std::uniform_real_distribution<> distrib(-1.0, 1.0); // Range: -1 to 1

        // Populate the layer with random values
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                layerMatrix[i][j] = distrib(gen);
            }
            layerBiases[i] = distrib(gen);
        }

        // append the layer
        this->weightsMatrices.push_back(layerMatrix);
        this->biasVectors.push_back(layerBiases);
    }
};

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
