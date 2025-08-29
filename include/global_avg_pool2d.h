#ifndef GLOBAL_AVG_POOL2D_H
#define GLOBAL_AVG_POOL2D_H

#include "tensor.h"

typedef struct {
    DoubleTensor* inp_buffer;
    int inp_buffer_size;
    int inp_buffer_capacity;
} TensorBuffer;

typedef struct {
    TensorBuffer inp_buffer;
    TensorBuffer inp_grad_buffer;
    char* name;
} GlobalAvgPool2d;

GlobalAvgPool2d global_avg_pool2d_create(void);
DoubleTensor global_avg_pool2d_forward(GlobalAvgPool2d* global_avg_pool2d, DoubleTensor inp);
DoubleTensor global_avg_pool2d_backward(GlobalAvgPool2d* global_avg_pool2d, DoubleTensor delta);
void global_avg_pool2d_free(GlobalAvgPool2d* global_avg_pool2d);

#endif