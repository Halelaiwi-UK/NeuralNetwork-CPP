#include "UtilityFunctions.h"

#include <algorithm>
#include <random>
#include <iostream>
#include <sstream>
#include <execution>

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
    // std::vector<double> result(actual.size());
    // std::transform(std::execution::par,
    //     actual.begin(), actual.end(), expected.begin(), result.begin(),
    //     [](const double val1, const double val2) {
    //             return 2 * (val1 - val2);
    //     });
    std::vector<double> result(actual.size());
    std::transform(std::execution::par,
    actual.begin(), actual.end(), expected.begin(), result.begin(),
    [&](const double val1, const double val2) {
        return (2 * (val1 - val2)) / actual.size(); // Normalize by vector size
    });
    return result;
}

double UtilityFunctions::SigmoidDerivative(const double value) {
    const double sigmoid = 1.0 / (1.0 + exp(-value));
    return sigmoid * (1 - sigmoid);
}
