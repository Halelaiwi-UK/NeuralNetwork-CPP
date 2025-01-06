#include "UtilityFunctions.h"

#include <algorithm>
#include <random>
#include <iostream>
#include <sstream>
#include <execution>
#include <fstream>

std::vector<double> UtilityFunctions::multiplyMatrixVector(
    const std::vector<std::vector<double>> &matrix,
    const std::vector<double> &vec)
{
    if (matrix.empty() || vec.empty() || matrix[0].size() != vec.size()) {
        size_t matrixRows = matrix.size();
        size_t matrixCols = matrix.empty() ? 0 : matrix[0].size();
        size_t vectorSize = vec.size();

        // Construct the error message
        std::ostringstream oss;
        oss << "Matrix and vector dimensions are incompatible. "
            << "Matrix dimensions: " << matrixRows << "x" << matrixCols << ", "
            << "Vector size: " << vectorSize << "x1";

        throw std::invalid_argument(oss.str());
    }

    // Matrix-vector multiplication
    const std::vector<double>::size_type rows(matrix.size());
    std::vector result(rows, 0.0);

    // Use std::transform with a parallel execution policy
    std::transform(std::execution::par,
        matrix.begin(), matrix.end(), result.begin(),
                   [&vec] // <- using
                   // in Lambda function
                   (const std::vector<double>& row) {
                       double sum = 0.0;
                       for (std::size_t j = 0; j < vec.size(); ++j) {
                           sum += row[j] * vec[j];
                       }
                       return sum;
                   });

    return result;
};

std::vector<double> UtilityFunctions::SigmoidVector(const std::vector<double>& vec) {
    std::vector<double> result(vec.size());
    std::transform(std::execution::par,
        vec.begin(), vec.end(), result.begin(),
        [](const double value) {
            return 1.0 / (1.0 + exp(-value));
        });
    return result;
}

std::vector<double> UtilityFunctions::ReluVector(const std::vector<double> &vec) {
    std::vector<double> result(vec.size());
    std::transform(std::execution::par,
        vec.begin(), vec.end(), result.begin(),
        [](const double value) {
            return value > 0 ? value : 0;
        });
    return result;
}

double UtilityFunctions::ReluDerivative(double value) {
    return value > 0 ? 1 : 0;
}

std::vector<double> UtilityFunctions::VectorAddition(const std::vector<double>& vec1, const std::vector<double>&  vec2) {
    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument(
            "Vector dimensions mismatch: vec1 size = " + std::to_string(vec1.size()) +
            ", vec2 size = " + std::to_string(vec2.size()));
    }
    std::vector<double> result(vec1.size());
    std::transform(std::execution::par,
        vec1.begin(), vec1.end(), vec2.begin(),  result.begin(),
        [](const double val1, const double val2) {
            return val1 + val2;
        });
    return result;
}

// Parallelized Mean Squared Error
std::vector<double> UtilityFunctions::MSE(const std::vector<double> &actual, const std::vector<double> &expected) {
    if (actual.size() != expected.size()) {
        throw std::invalid_argument(
            "Vector dimensions mismatch: arg1 size = " + std::to_string(actual.size()) +
            ", arg2 size = " + std::to_string(expected.size()));
    }
    std::vector<double> result(actual.size());
    std::transform(std::execution::par,
        actual.begin(), actual.end(), expected.begin(), result.begin(),
        [](const double val1, const double val2) {
                return std::pow(val1 - val2, 2);
        });
    return result;
}

std::vector<double> UtilityFunctions::MSE_derivative(const std::vector<double> &actual,
    const std::vector<double> &expected) {
    if (actual.size() != expected.size()) {
        throw std::invalid_argument(
            "Vector dimensions mismatch: arg1 size = " + std::to_string(actual.size()) +
            ", arg2 size = " + std::to_string(expected.size()));
    }
    std::vector<double> result(actual.size());
    std::transform(std::execution::par,
    actual.begin(), actual.end(), expected.begin(), result.begin(),
    [&](const double val1, const double val2) {
        return (2 * (val1 - val2)) / static_cast<double>(actual.size()); // Normalize by vector size
    });
    return result;
}

double UtilityFunctions::SigmoidDerivative(const double value) {
    const double sigmoid = 1.0 / (1.0 + exp(-value));
    return sigmoid * (1 - sigmoid);
}

std::vector<double> oneHotEncode(int label, int numClasses = 10) {
    if (label < 0 || label >= numClasses) {
        throw std::invalid_argument("Invalid label value for one-hot encoding.");
    }

    std::vector<double> oneHot(numClasses, 0);
    oneHot[label] = 1;
    return oneHot;
}

std::vector<ImageData> UtilityFunctions::loadData(const std::string& filename, bool isTest) {
    std::vector<ImageData> dataset; // Vector to store all image data
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    // Skip the first line (header)
    std::string header;
    if (!std::getline(file, header)) {
        throw std::runtime_error("File is empty or unable to read header: " + filename);
    }


    std::string line;
    while (std::getline(file, line)) {
        std::istringstream lineStream(line);
        ImageData imageData;

        // Read the label (first value in the line) ignore if testing data
        std::string token;
        if (!isTest) {
            if (std::getline(lineStream, token, ',')) {
                int labelValue = std::stoi(token); // Convert label to integer
                imageData.label = oneHotEncode(labelValue);
            }
        }

        // Read pixel values (remaining values in the line)
        while (std::getline(lineStream, token, ',')) {
            if (!token.empty()) {
                try {
                    double pixelValue = std::stod(token); // Convert pixel to double
                    imageData.pixels.push_back(pixelValue);
                } catch (const std::invalid_argument& e) {
                    std::cerr << "Invalid pixel value: " << token << ". Error: " << e.what() << std::endl;
                    throw;
                }
            }
        }
        dataset.push_back(imageData); // Add the parsed data to the dataset
    }

    file.close();
    return dataset;
}

std::vector<double> UtilityFunctions::Softmax(const std::vector<double>& input) {
    std::vector<double> output(input.size());
    double maxInput = *std::ranges::max_element(input); // Shift by max value
    double sum = 0.0;

    for (double val : input) {
        sum += std::exp(val - maxInput);
    }

    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = std::exp(input[i] - maxInput) / sum;
    }

    return output;
}

double UtilityFunctions::CrossEntropy(const std::vector<double>& predicted, const std::vector<double>& actual) {
    double loss = 0.0;
    for (size_t i = 0; i < predicted.size(); ++i) {
        double clamped_pred = std::clamp(predicted[i], 1e-9, 1.0); // Clamp predictions
        loss -= actual[i] * std::log(clamped_pred);
    }
    return loss;
}