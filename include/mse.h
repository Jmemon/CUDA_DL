#ifndef MSE_H
#define MSE_H

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
} MSE;

MSE mse_create(void);
DoubleTensor mse_forward(MSE* mse, DoubleTensor inp, DoubleTensor target);
DoubleTensor mse_backward(MSE* mse, DoubleTensor preds);
void mse_free(MSE* mse);

#endif