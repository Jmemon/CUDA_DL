#ifndef LINEAR_H
#define LINEAR_H

#include "tensor.h"
#include "module.h"

// Linear module functions
Module* linear_create(int features_in, int features_out);
void linear_forward_impl(Module* module, DoubleTensor* input, DoubleTensor* output);
void linear_backward_impl(Module* module, DoubleTensor* grad_output, DoubleTensor* grad_input);

#endif