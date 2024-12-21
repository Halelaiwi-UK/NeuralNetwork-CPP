#include <algorithm>
#include <iostream>
#include <NeuralNetwork.h>
#include <UtilityFunctions.h>
#include <fstream>
void saveAsPGM(const std::vector<double>& pixels, const std::string& filename, int width, int height) {
    if (pixels.size() != width * height) {
        throw std::invalid_argument("Pixel size does not match image dimensions");
    }

    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    // Write PGM header
    file << "P2\n" << width << " " << height << "\n255\n";

    // Write pixel data
    for (size_t i = 0; i < pixels.size(); ++i) {
        file << static_cast<int>(pixels[i]) << " ";
        if ((i + 1) % width == 0) {
            file << "\n"; // Newline after each row
        }
    }

    file.close();
}

int main() {
    std::string trainDataPath = "train.csv"; // Replace with your file path
    std::vector<ImageData> trainData = UtilityFunctions::loadData(trainDataPath, false);
    std::string testDataPath = "test.csv"; // Replace with your file path
    std::vector<ImageData> testData = UtilityFunctions::loadData(testDataPath, true);

    // Separate train data into two vectors: one for pixels, one for labels
    std::vector<std::vector<double>> trainPixels;
    std::vector<std::vector<double>> trainLabels;

    for (const auto& data : trainData) {
        trainPixels.push_back(data.pixels); // Extract pixels
        trainLabels.push_back(data.label); // Extract labels
    }
    auto network = NeuralNetwork(trainPixels[0].size()); // Initialize network and input layer
    network.setLearningRate(0.001);

    network.add_layer(128); // hidden layer
    network.add_layer(128);
    network.add_layer(trainLabels[0].size()); // Output layer
    network.train(trainPixels, trainLabels, 25);

    // Separate train data into two vectors: one for pixels, one for labels
    std::vector<std::vector<double>> testPixels;
    std::vector<std::vector<double>> testLabels;
    for (const auto& data : testData) {
        testPixels.push_back(data.pixels); // Extract pixels
        testLabels.push_back(data.label); // Extract labels
    }

    for (int i = 0; i < 5; i++) {
        auto result = network.predict(testPixels[i]);
        std::cout << "[";
        for (auto& value : result) {
            std::cout << value << ", ";
        }
        std::cout << "]" << std::endl;
        // Find the index of the maximum element
        auto maxElementIt = std::max_element(result.begin(), result.end());
        int maxIndex = std::distance(result.begin(), maxElementIt);
        std::cout << "Prediction : " << maxIndex << std::endl;

        std::string filename = "testCaseNumber" + std::to_string(i) + ".pgm";
        saveAsPGM(testPixels[i], filename, 28, 28);
    }

    return 0;
}
