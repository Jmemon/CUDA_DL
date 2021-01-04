#ifndef ACTIVATION_CUH
#define ACTIVATION_CUH

#include <vector>

// More detailed information about each function in src/Activation.cu

// All of these functions expect a one dimensional row of threads

// calls binaryStep cuda kernel
std::vector<double> binaryStepGPU(std::vector<double>& z);

// binary step deriv is weird, it seems like it would always be 0, except at 0
// but seems like it would make backprop stop working

// calls sigmoid cuda kernel if !diff, otherwise calls sigmoid_prime 
std::vector<double> sigmoidGPU(std::vector<double>& z, bool diff = false);

// calls relu cuda kernel if !diff, otherwise calls relu_prime 
std::vector<double> reluGPU(std::vector<double>& z, bool diff = false);

// calls leakyRelu cuda kernel if !diff, otherwise calls leakyRelu_prime 
std::vector<double> leakyReluGPU(std::vector<double>& z, bool diff = false);

// calls exponential cuda kernel if !diff, otherwise calls exponential, since deriv(e^x) = e^x
std::vector<double> exponentialGPU(std::vector<double>& z, bool diff = false);

#endif // ACTIVATION_CUH
