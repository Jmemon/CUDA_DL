#ifndef ACTIVATION_CUH
#define ACTIVATION_CUH

#include <vector>

// All of these functions expect a one dimensional row of threads

std::vector<double> binaryStepGPU(std::vector<double>& z);

// binary step deriv is weird, it seems like it would always be 0, except at 0
// but seems like backprop would stop working

std::vector<double> sigmoidGPU(std::vector<double>& z, bool diff = false);

std::vector<double> reluGPU(std::vector<double>& z, bool diff = false);

std::vector<double> leakyReluGPU(std::vector<double>& z, bool diff = false);

#endif // ACTIVATION_CUH
