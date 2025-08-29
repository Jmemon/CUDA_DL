#ifndef RELU_H
#define RELU_H

#include "tensor.h"
#include "module.h"

// ReLU module functions (no data needed)
Module* relu_create(void);
void relu_forward_impl(Module* module, DoubleTensor* input, DoubleTensor* output);
void relu_backward_impl(Module* module, DoubleTensor* grad_output, DoubleTensor* grad_input);

#endif