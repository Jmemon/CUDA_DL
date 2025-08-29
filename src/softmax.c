#include "../include/softmax.h"
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

Softmax softmax_create(void) {
    Softmax softmax;
    tensor_buffer_init(&softmax.inp_buffer);
    tensor_buffer_init(&softmax.inp_grad_buffer);
    
    softmax.name = (char*)malloc(8 * sizeof(char));
    strcpy(softmax.name, "softmax");
    
    return softmax;
}

DoubleTensor softmax_forward(Softmax* softmax, DoubleTensor inp) {
    DoubleTensor inp_copy;
    inp_copy.dims = (int*)malloc(inp.num_dims * sizeof(int));
    memcpy(inp_copy.dims, inp.dims, inp.num_dims * sizeof(int));
    inp_copy.num_dims = inp.num_dims;
    inp_copy.num_elem = inp.num_elem;
    inp_copy.elems = (double*)malloc(inp.num_elem * sizeof(double));
    memcpy(inp_copy.elems, inp.elems, inp.num_elem * sizeof(double));
    
    tensor_buffer_add(&softmax->inp_buffer, inp_copy);
    
    DoubleTensor result = double_tensor_alloc(inp.dims, inp.num_dims);
    
    double max_val = inp.elems[0];
    for (int i = 1; i < inp.num_elem; i++) {
        if (inp.elems[i] > max_val) {
            max_val = inp.elems[i];
        }
    }
    
    double sum = 0.0;
    for (int i = 0; i < inp.num_elem; i++) {
        result.elems[i] = exp(inp.elems[i] - max_val);
        sum += result.elems[i];
    }
    
    for (int i = 0; i < inp.num_elem; i++) {
        result.elems[i] /= sum;
    }
    
    return result;
}

DoubleTensor softmax_backward(Softmax* softmax, DoubleTensor delta) {
    if (softmax->inp_buffer.inp_buffer_size == 0) {
        DoubleTensor empty = {0};
        return empty;
    }
    
    DoubleTensor inp_grad = double_tensor_alloc(delta.dims, delta.num_dims);
    
    for (int i = 0; i < delta.num_elem; i++) {
        inp_grad.elems[i] = delta.elems[i];
    }
    
    tensor_buffer_add(&softmax->inp_grad_buffer, inp_grad);
    
    return inp_grad;
}

void softmax_free(Softmax* softmax) {
    tensor_buffer_free(&softmax->inp_buffer);
    tensor_buffer_free(&softmax->inp_grad_buffer);
    
    if (softmax->name) {
        free(softmax->name);
        softmax->name = NULL;
    }
}