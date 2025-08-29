#ifndef CROSS_ENTROPY_H
#define CROSS_ENTROPY_H

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
} CrossEntropy;

CrossEntropy cross_entropy_create(void);
DoubleTensor cross_entropy_forward(CrossEntropy* cross_entropy, DoubleTensor inp, DoubleTensor target);
DoubleTensor cross_entropy_backward(CrossEntropy* cross_entropy, DoubleTensor preds);
void cross_entropy_free(CrossEntropy* cross_entropy);

#endif