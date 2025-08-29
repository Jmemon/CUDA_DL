#ifndef SOFTMAX_H
#define SOFTMAX_H

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
} Softmax;

Softmax softmax_create(void);
DoubleTensor softmax_forward(Softmax* softmax, DoubleTensor inp);
DoubleTensor softmax_backward(Softmax* softmax, DoubleTensor delta);
void softmax_free(Softmax* softmax);

#endif