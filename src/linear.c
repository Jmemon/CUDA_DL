#include "../include/linear.h"
#include <stdlib.h>
#include <string.h>

Module* linear_create(int features_in, int features_out) {
    // Define parameter names for linear layer
    char* param_names[] = {"weight", "bias"};
    char* hyperparam_names[] = {"features_in", "features_out"};
    
    Module* module = module_create("linear", param_names, 2, hyperparam_names, 2);
    if (!module) return NULL;
    
    // Set hyperparameters
    module_set_hyperparam_value(module, "features_in", (double)features_in);
    module_set_hyperparam_value(module, "features_out", (double)features_out);
    
    // Create and initialize parameters
    // Weight tensor: [features_out, features_in]
    int weight_dims[2] = {features_out, features_in};
    DoubleTensor* weight = (DoubleTensor*)malloc(sizeof(DoubleTensor));
    *weight = double_tensor_alloc(weight_dims, 2);
    double_tensor_random_init(weight);
    module_set_param_value(module, "weight", weight);
    
    // Bias tensor: [features_out]
    int bias_dims[1] = {features_out};
    DoubleTensor* bias = (DoubleTensor*)malloc(sizeof(DoubleTensor));
    *bias = double_tensor_alloc(bias_dims, 1);
    double_tensor_zero_init(bias);
    module_set_param_value(module, "bias", bias);
    
    // Initialize gradients
    DoubleTensor* weight_grad = (DoubleTensor*)malloc(sizeof(DoubleTensor));
    *weight_grad = double_tensor_alloc(weight_dims, 2);
    double_tensor_zero_init(weight_grad);
    module_set_param_grad(module, "weight", weight_grad);
    
    DoubleTensor* bias_grad = (DoubleTensor*)malloc(sizeof(DoubleTensor));
    *bias_grad = double_tensor_alloc(bias_dims, 1);
    double_tensor_zero_init(bias_grad);
    module_set_param_grad(module, "bias", bias_grad);
    
    // Set function pointers
    module_set_functions(module, linear_forward_impl, linear_backward_impl);
    
    return module;
}

void linear_forward_impl(Module* module, DoubleTensor* input, DoubleTensor* output) {
    if (!module || !input || !output) return;
    
    DoubleTensor* weight = module_get_param_value(module, "weight");
    DoubleTensor* bias = module_get_param_value(module, "bias");
    
    if (!weight || !bias) return;
    
    // Store input for backward pass
    DoubleTensor* input_copy = (DoubleTensor*)malloc(sizeof(DoubleTensor));
    input_copy->dims = (int*)malloc(input->num_dims * sizeof(int));
    memcpy(input_copy->dims, input->dims, input->num_dims * sizeof(int));
    input_copy->num_dims = input->num_dims;
    input_copy->num_elem = input->num_elem;
    input_copy->elems = (double*)malloc(input->num_elem * sizeof(double));
    memcpy(input_copy->elems, input->elems, input->num_elem * sizeof(double));
    module_set_input_buffer(module, "input", input_copy);
    
    // Perform matrix multiplication: output = input @ weight.T + bias
    *output = double_tensor_matmul(input, weight);
    
    if (output->elems == NULL) {
        return;
    }
    
    // Add bias
    int features_out = (int)module_get_hyperparam_value(module, "features_out");
    for (int i = 0; i < output->num_elem; i++) {
        output->elems[i] += bias->elems[i % features_out];
    }
}

void linear_backward_impl(Module* module, DoubleTensor* grad_output, DoubleTensor* grad_input) {
    if (!module || !grad_output) return;
    
    DoubleTensor* input = module_get_input_buffer(module, "input");
    DoubleTensor* weight = module_get_param_value(module, "weight");
    
    if (!input || !weight) return;
    
    int features_in = (int)module_get_hyperparam_value(module, "features_in");
    int features_out = (int)module_get_hyperparam_value(module, "features_out");
    
    // Compute weight gradient: weight_grad = grad_output.T @ input
    DoubleTensor* weight_grad = module_get_param_grad(module, "weight");
    if (weight_grad) {
        // Reset gradient
        double_tensor_zero_init(weight_grad);
        
        // Compute gradient for weights
        for (int i = 0; i < features_out; i++) {
            for (int j = 0; j < features_in; j++) {
                weight_grad->elems[i * features_in + j] += 
                    grad_output->elems[i] * input->elems[j];
            }
        }
    }
    
    // Compute bias gradient: bias_grad = grad_output
    DoubleTensor* bias_grad = module_get_param_grad(module, "bias");
    if (bias_grad) {
        // Reset gradient
        double_tensor_zero_init(bias_grad);
        
        // Copy grad_output to bias_grad
        memcpy(bias_grad->elems, grad_output->elems, features_out * sizeof(double));
    }
    
    // Compute input gradient: grad_input = grad_output @ weight
    if (grad_input) {
        // Initialize grad_input
        int input_dims[1] = {features_in};
        *grad_input = double_tensor_alloc(input_dims, 1);
        double_tensor_zero_init(grad_input);
        
        // Compute gradient for input
        for (int j = 0; j < features_in; j++) {
            for (int i = 0; i < features_out; i++) {
                grad_input->elems[j] += weight->elems[i * features_in + j] * grad_output->elems[i];
            }
        }
        
        // Store input gradient in buffer
        DoubleTensor* grad_input_copy = (DoubleTensor*)malloc(sizeof(DoubleTensor));
        grad_input_copy->dims = (int*)malloc(grad_input->num_dims * sizeof(int));
        memcpy(grad_input_copy->dims, grad_input->dims, grad_input->num_dims * sizeof(int));
        grad_input_copy->num_dims = grad_input->num_dims;
        grad_input_copy->num_elem = grad_input->num_elem;
        grad_input_copy->elems = (double*)malloc(grad_input->num_elem * sizeof(double));
        memcpy(grad_input_copy->elems, grad_input->elems, grad_input->num_elem * sizeof(double));
        module_set_input_grad_buffer(module, "input", grad_input_copy);
    }
}

