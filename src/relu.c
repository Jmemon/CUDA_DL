#include "../include/relu.h"
#include <stdlib.h>
#include <string.h>

Module* relu_create(void) {
    // ReLU has no parameters or hyperparameters
    Module* module = module_create("relu", NULL, 0, NULL, 0);
    if (!module) return NULL;
    
    // Set function pointers
    module_set_functions(module, relu_forward_impl, relu_backward_impl);
    
    return module;
}

void relu_forward_impl(Module* module, DoubleTensor* input, DoubleTensor* output) {
    if (!module || !input || !output) return;
    
    // Store input for backward pass
    DoubleTensor* input_copy = (DoubleTensor*)malloc(sizeof(DoubleTensor));
    input_copy->dims = (int*)malloc(input->num_dims * sizeof(int));
    memcpy(input_copy->dims, input->dims, input->num_dims * sizeof(int));
    input_copy->num_dims = input->num_dims;
    input_copy->num_elem = input->num_elem;
    input_copy->elems = (double*)malloc(input->num_elem * sizeof(double));
    memcpy(input_copy->elems, input->elems, input->num_elem * sizeof(double));
    module_set_input_buffer(module, "input", input_copy);
    
    // Allocate output tensor
    *output = double_tensor_alloc(input->dims, input->num_dims);
    
    // Apply ReLU activation: max(0, x)
    for (int i = 0; i < input->num_elem; i++) {
        output->elems[i] = input->elems[i] > 0.0 ? input->elems[i] : 0.0;
    }
}

void relu_backward_impl(Module* module, DoubleTensor* grad_output, DoubleTensor* grad_input) {
    if (!module || !grad_output) return;
    
    DoubleTensor* input = module_get_input_buffer(module, "input");
    if (!input) return;
    
    // Compute input gradient
    if (grad_input) {
        *grad_input = double_tensor_alloc(grad_output->dims, grad_output->num_dims);
        
        // ReLU derivative: grad_input[i] = grad_output[i] if input[i] > 0, else 0
        for (int i = 0; i < grad_output->num_elem; i++) {
            grad_input->elems[i] = input->elems[i] > 0.0 ? grad_output->elems[i] : 0.0;
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

