#include "../include/mse.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void tensor_buffer_init(TensorBuffer* buffer) {
    buffer->inp_buffer = NULL;
    buffer->inp_buffer_size = 0;
    buffer->inp_buffer_capacity = 0;
}

static void tensor_buffer_add(TensorBuffer* buffer, DoubleTensor tensor) {
    if (buffer->inp_buffer_size >= buffer->inp_buffer_capacity) {
        buffer->inp_buffer_capacity = buffer->inp_buffer_capacity == 0 ? 1 : buffer->inp_buffer_capacity * 2;
        buffer->inp_buffer = (DoubleTensor*)realloc(buffer->inp_buffer, buffer->inp_buffer_capacity * sizeof(DoubleTensor));
    }
    buffer->inp_buffer[buffer->inp_buffer_size++] = tensor;
}

static void tensor_buffer_free(TensorBuffer* buffer) {
    for (int i = 0; i < buffer->inp_buffer_size; i++) {
        double_tensor_free(&buffer->inp_buffer[i]);
    }
    if (buffer->inp_buffer) {
        free(buffer->inp_buffer);
    }
    buffer->inp_buffer = NULL;
    buffer->inp_buffer_size = 0;
    buffer->inp_buffer_capacity = 0;
}

MSE mse_create(void) {
    MSE mse;
    
    tensor_buffer_init(&mse.inp_buffer);
    tensor_buffer_init(&mse.inp_grad_buffer);
    
    mse.name = (char*)malloc(4 * sizeof(char));
    strcpy(mse.name, "mse");
    
    return mse;
}

DoubleTensor mse_forward(MSE* mse, DoubleTensor inp, DoubleTensor target) {
    DoubleTensor inp_copy;
    inp_copy.dims = (int*)malloc(inp.num_dims * sizeof(int));
    memcpy(inp_copy.dims, inp.dims, inp.num_dims * sizeof(int));
    inp_copy.num_dims = inp.num_dims;
    inp_copy.num_elem = inp.num_elem;
    inp_copy.elems = (double*)malloc(inp.num_elem * sizeof(double));
    memcpy(inp_copy.elems, inp.elems, inp.num_elem * sizeof(double));
    
    tensor_buffer_add(&mse->inp_buffer, inp_copy);
    
    if (inp.num_elem != target.num_elem) {
        DoubleTensor empty = {0};
        return empty;
    }
    
    int result_dims[1] = {1};
    DoubleTensor result = double_tensor_alloc(result_dims, 1);
    
    double loss = 0.0;
    for (int i = 0; i < inp.num_elem; i++) {
        double diff = inp.elems[i] - target.elems[i];
        loss += diff * diff;
    }
    loss /= inp.num_elem;
    
    result.elems[0] = loss;
    return result;
}

DoubleTensor mse_backward(MSE* mse, DoubleTensor preds) {
    if (mse->inp_buffer.inp_buffer_size == 0) {
        DoubleTensor empty = {0};
        return empty;
    }
    
    DoubleTensor inp_grad = double_tensor_alloc(preds.dims, preds.num_dims);
    
    for (int i = 0; i < preds.num_elem; i++) {
        inp_grad.elems[i] = 2.0 * preds.elems[i] / preds.num_elem;
    }
    
    tensor_buffer_add(&mse->inp_grad_buffer, inp_grad);
    
    return inp_grad;
}

void mse_free(MSE* mse) {
    tensor_buffer_free(&mse->inp_buffer);
    tensor_buffer_free(&mse->inp_grad_buffer);
    
    if (mse->name) {
        free(mse->name);
        mse->name = NULL;
    }
}