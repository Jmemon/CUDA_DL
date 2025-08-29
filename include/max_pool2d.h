#ifndef MAX_POOL2D_H
#define MAX_POOL2D_H

#include "tensor.h"

typedef struct {
    DoubleTensor* inp_buffer;
    int inp_buffer_size;
    int inp_buffer_capacity;
} TensorBuffer;

typedef struct {
    int kernel_size;
    TensorBuffer inp_buffer;
    TensorBuffer inp_grad_buffer;
    char* name;
} MaxPool2d;

MaxPool2d max_pool2d_create(int kernel_size);
DoubleTensor max_pool2d_forward(MaxPool2d* max_pool2d, DoubleTensor inp);
DoubleTensor max_pool2d_backward(MaxPool2d* max_pool2d, DoubleTensor delta);
void max_pool2d_free(MaxPool2d* max_pool2d);

#endif