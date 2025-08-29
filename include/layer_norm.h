#ifndef LAYER_NORM_H
#define LAYER_NORM_H

#include "tensor.h"

typedef struct {
    DoubleTensor* inp_buffer;
    int inp_buffer_size;
    int inp_buffer_capacity;
} TensorBuffer;

typedef struct {
    int features;
    TensorBuffer inp_buffer;
    TensorBuffer inp_grad_buffer;
    char* name;
} LayerNorm;

LayerNorm layer_norm_create(int features);
DoubleTensor layer_norm_forward(LayerNorm* layer_norm, DoubleTensor inp);
DoubleTensor layer_norm_backward(LayerNorm* layer_norm, DoubleTensor delta);
void layer_norm_free(LayerNorm* layer_norm);

#endif