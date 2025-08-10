#ifndef ACTIVATION_CUH
#define ACTIVATION_CUH

#include <vector>

// More detailed information about each function in src/Activation.cu

// All of these functions expect a one dimensional row of threads

// calls sigmoid cuda kernel if !diff, otherwise calls sigmoid_prime 
std::vector<double> sigmoidGPU(std::vector<double>& z, bool diff = false);

// calls tanh cuda kernel if !diff, otherwise calls tanh_prime 
std::vector<double> tanhGPU(std::vector<double>& z, bool diff = false);

// calls relu cuda kernel if !diff, otherwise calls relu_prime 
std::vector<double> reluGPU(std::vector<double>& z, bool diff = false);

// calls leakyRelu cuda kernel if !diff, otherwise calls leakyRelu_prime 
std::vector<double> leakyReluGPU(std::vector<double>& z, bool diff = false);


#endif // ACTIVATION_CUH
