#ifndef CONV2D_H
#define CONV2D_H

#include "tensor.h"
#include "module.h"

// Conv2d module functions
Module* conv2d_create(int in_channels, int out_channels, int kernel_size, int stride, int padding);
void conv2d_forward_impl(Module* module, DoubleTensor* input, DoubleTensor* output);
void conv2d_backward_impl(Module* module, DoubleTensor* grad_output, DoubleTensor* grad_input);

#endif