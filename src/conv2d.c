#include "../include/conv2d.h"
#include <stdlib.h>
#include <string.h>

Module* conv2d_create(int in_channels, int out_channels, int kernel_size, int stride, int padding) {
    // Define parameter names for conv2d layer
    char* param_names[] = {"kernel"};
    char* hyperparam_names[] = {"in_channels", "out_channels", "kernel_size", "stride", "padding"};
    
    Module* module = module_create("conv2d", param_names, 1, hyperparam_names, 5);
    if (!module) return NULL;
    
    // Set hyperparameters
    module_set_hyperparam_value(module, "in_channels", (double)in_channels);
    module_set_hyperparam_value(module, "out_channels", (double)out_channels);
    module_set_hyperparam_value(module, "kernel_size", (double)kernel_size);
    module_set_hyperparam_value(module, "stride", (double)stride);
    module_set_hyperparam_value(module, "padding", (double)padding);
    
    // Create and initialize kernel parameter
    // Kernel tensor: [out_channels, in_channels, kernel_size, kernel_size]
    int kernel_dims[4] = {out_channels, in_channels, kernel_size, kernel_size};
    DoubleTensor* kernel = (DoubleTensor*)malloc(sizeof(DoubleTensor));
    *kernel = double_tensor_alloc(kernel_dims, 4);
    double_tensor_random_init(kernel);
    module_set_param_value(module, "kernel", kernel);
    
    // Initialize gradient
    DoubleTensor* kernel_grad = (DoubleTensor*)malloc(sizeof(DoubleTensor));
    *kernel_grad = double_tensor_alloc(kernel_dims, 4);
    double_tensor_zero_init(kernel_grad);
    module_set_param_grad(module, "kernel", kernel_grad);
    
    // Set function pointers
    module_set_functions(module, conv2d_forward_impl, conv2d_backward_impl);
    
    return module;
}

void conv2d_forward_impl(Module* module, DoubleTensor* input, DoubleTensor* output) {
    if (!module || !input || !output) return;
    
    DoubleTensor* kernel = module_get_param_value(module, "kernel");
    if (!kernel) return;
    
    int in_channels = (int)module_get_hyperparam_value(module, "in_channels");
    int out_channels = (int)module_get_hyperparam_value(module, "out_channels");
    int kernel_size = (int)module_get_hyperparam_value(module, "kernel_size");
    int stride = (int)module_get_hyperparam_value(module, "stride");
    int padding = (int)module_get_hyperparam_value(module, "padding");
    
    // Store input for backward pass
    DoubleTensor* input_copy = (DoubleTensor*)malloc(sizeof(DoubleTensor));
    input_copy->dims = (int*)malloc(input->num_dims * sizeof(int));
    memcpy(input_copy->dims, input->dims, input->num_dims * sizeof(int));
    input_copy->num_dims = input->num_dims;
    input_copy->num_elem = input->num_elem;
    input_copy->elems = (double*)malloc(input->num_elem * sizeof(double));
    memcpy(input_copy->elems, input->elems, input->num_elem * sizeof(double));
    module_set_input_buffer(module, "input", input_copy);
    
    // Validate input dimensions
    if (input->num_dims != 3 && input->num_dims != 4) {
        return;
    }
    
    int batch_size = (input->num_dims == 4) ? input->dims[0] : 1;
    int in_channels = (input->num_dims == 4) ? input->dims[1] : input->dims[0];
    int in_height = (input->num_dims == 4) ? input->dims[2] : input->dims[1];
    int in_width = (input->num_dims == 4) ? input->dims[3] : input->dims[2];
    
    // Calculate output dimensions
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    // Allocate output tensor
    int result_dims[4] = {batch_size, out_channels, out_height, out_width};
    *output = double_tensor_alloc(result_dims, input->num_dims == 4 ? 4 : 3);
    double_tensor_zero_init(output);
    
    // Perform convolution
    for (int b = 0; b < batch_size; b++) {
        for (int oc = 0; oc < out_channels; oc++) {
            for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                    double sum = 0.0;
                    
                    for (int ic = 0; ic < in_channels; ic++) {
                        for (int kh = 0; kh < kernel_size; kh++) {
                            for (int kw = 0; kw < kernel_size; kw++) {
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;
                                
                                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                    int inp_idx, kernel_idx;
                                    if (input->num_dims == 4) {
                                        inp_idx = b * in_channels * in_height * in_width + 
                                                 ic * in_height * in_width + ih * in_width + iw;
                                    } else {
                                        inp_idx = ic * in_height * in_width + ih * in_width + iw;
                                    }
                                    
                                    kernel_idx = oc * in_channels * kernel_size * kernel_size + 
                                               ic * kernel_size * kernel_size + 
                                               kh * kernel_size + kw;
                                    
                                    sum += input->elems[inp_idx] * kernel->elems[kernel_idx];
                                }
                            }
                        }
                    }
                    
                    int result_idx;
                    if (output->num_dims == 4) {
                        result_idx = b * out_channels * out_height * out_width + 
                                    oc * out_height * out_width + oh * out_width + ow;
                    } else {
                        result_idx = oc * out_height * out_width + oh * out_width + ow;
                    }
                    
                    output->elems[result_idx] = sum;
                }
            }
        }
    }
}

void conv2d_backward_impl(Module* module, DoubleTensor* grad_output, DoubleTensor* grad_input) {
    if (!module || !grad_output) return;
    
    DoubleTensor* input = module_get_input_buffer(module, "input");
    DoubleTensor* kernel = module_get_param_value(module, "kernel");
    
    if (!input || !kernel) return;
    
    // Initialize kernel gradient
    DoubleTensor* kernel_grad = module_get_param_grad(module, "kernel");
    if (kernel_grad) {
        double_tensor_zero_init(kernel_grad);
        // TODO: Implement kernel gradient computation for conv2d
    }
    
    // Initialize input gradient  
    if (grad_input) {
        *grad_input = double_tensor_alloc(input->dims, input->num_dims);
        double_tensor_zero_init(grad_input);
        // TODO: Implement input gradient computation for conv2d
        
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

