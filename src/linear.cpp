#include "../include/linear.h"
#include "../include/Matrix.cuh"
#include <random>
#include <cmath>

Linear::Linear(int in_size, int out_size) : in_size(in_size), out_size(out_size) {
    initialize_parameters();
}

void Linear::initialize_parameters() {
    add_parameter_group("weights", in_size * out_size);
    add_parameter_group("bias", out_size);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, sqrt(2.0 / in_size));
    
    // Initialize weights with Xavier/He initialization
    for (int i = 0; i < in_size * out_size; i++) {
        parameters[i] = dist(gen);
    }
    
    // Initialize biases to zero
    for (int i = in_size * out_size; i < parameters.size(); i++) {
        parameters[i] = 0.0;
    }
}

std::vector<double> Linear::forward(std::vector<double> &x) {
    input_buffer = x;
    
    std::vector<double> weights = get_weights();
    std::vector<double> bias = get_bias();
    
    // y = x * W^T + b
    // x is (1, in_size), W is (out_size, in_size), so W^T is (in_size, out_size)
    std::vector<double> output = matMulGPU(x, weights, 1, in_size, out_size);
    
    // Add bias: output = output + bias
    output = matAddGPU(output, bias, 1, out_size);
    
    return output;
}

std::vector<double> Linear::backward(std::vector<double> &dL_dz) {
    input_gradients.resize(in_size);
    std::fill(parameter_gradients.begin(), parameter_gradients.end(), 0.0);
    
    std::vector<double> weights = get_weights();
    
    // Compute gradients w.r.t. weights: dL/dW = dL_dz^T * input
    // dL_dz is (1, out_size), input is (1, in_size)
    // We need (out_size, in_size) result
    std::vector<double> dL_dz_T = matTransGPU(dL_dz, 1, out_size);  // (out_size, 1)
    std::vector<double> weight_grads = matMulGPU(dL_dz_T, input_buffer, out_size, 1, in_size);
    
    // Copy weight gradients to parameter_gradients
    for (int i = 0; i < weight_grads.size(); i++) {
        parameter_gradients[i] = weight_grads[i];
    }
    
    // Bias gradients: dL/db = dL_dz
    for (int i = 0; i < out_size; i++) {
        parameter_gradients[in_size * out_size + i] = dL_dz[i];
    }
    
    // Compute gradients w.r.t. inputs: dL/dx = dL_dz * W
    // dL_dz is (1, out_size), W is (out_size, in_size)
    input_gradients = matMulGPU(dL_dz, weights, 1, out_size, in_size);
    
    return input_gradients;
}

std::vector<double> Linear::get_weights() const {
    return get_parameter_group("weights");
}

std::vector<double> Linear::get_bias() const {
    return get_parameter_group("bias");
}

std::vector<double> Linear::get_weight_gradients() const {
    return get_parameter_gradient_group("weights");
}

std::vector<double> Linear::get_bias_gradients() const {
    return get_parameter_gradient_group("bias");
}