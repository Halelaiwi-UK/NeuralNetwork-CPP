//
// Created by asus-pc on 12/21/2024.
//

#ifndef UTILITYFUNCTIONS_H
#define UTILITYFUNCTIONS_H
#include <vector>
#include <string>

struct ImageData {
    std::vector<double> label;        // One-hot encoded label (size 10)
    std::vector<double> pixels;       // Pixel values (0-255)
};

class UtilityFunctions {
public:
    static std::vector<double> multiplyMatrixVector(const std::vector<std::vector<double>>& matrix, const std::vector<double>& vec);
    static std::vector<double> SigmoidVector(const std::vector<double>& vec);
    static std::vector<double> VectorAddition(const std::vector<double>& vec1, const std::vector<double>&  vec2);
    static std::vector<double> MSE(const std::vector<double>& actual, const std::vector<double>&  expected);
    //  double sum_vector = std::reduce(std::execution::seq, vec.begin(), vec.end(), 0.0);
    static std::vector<double> MSE_derivative(const std::vector<double>& actual, const std::vector<double>&  expected);
    static double SigmoidDerivative(double value);
    static std::vector<ImageData> loadData(const std::string& filename, bool isTest);
};



#endif //UTILITYFUNCTIONS_H
