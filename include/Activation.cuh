#ifndef ACTIVATION_CUH
#define ACTIVATION_CUH

#include <vector>

// All of these functions expect a one dimensional row of threads

std::vector<double> binaryStepGPU(std::vector<double>& z);

std::vector<double> sigmoidGPU(std::vector<double>& z);

std::vector<double> reluGPU(std::vector<double>& z);

std::vector<double> leakyReluGPU(std::vector<double>& z);

#endif // ACTIVATION_CUH
