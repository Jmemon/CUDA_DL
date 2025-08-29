#include "../include/cross_entropy.h"
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

CrossEntropy cross_entropy_create(void) {
    CrossEntropy cross_entropy;
    
    tensor_buffer_init(&cross_entropy.inp_buffer);
    tensor_buffer_init(&cross_entropy.inp_grad_buffer);
    
    cross_entropy.name = (char*)malloc(14 * sizeof(char));
    strcpy(cross_entropy.name, "cross_entropy");
    
    return cross_entropy;
}

DoubleTensor cross_entropy_forward(CrossEntropy* cross_entropy, DoubleTensor inp, DoubleTensor target) {
    DoubleTensor inp_copy;
    inp_copy.dims = (int*)malloc(inp.num_dims * sizeof(int));
    memcpy(inp_copy.dims, inp.dims, inp.num_dims * sizeof(int));
    inp_copy.num_dims = inp.num_dims;
    inp_copy.num_elem = inp.num_elem;
    inp_copy.elems = (double*)malloc(inp.num_elem * sizeof(double));
    memcpy(inp_copy.elems, inp.elems, inp.num_elem * sizeof(double));
    
    tensor_buffer_add(&cross_entropy->inp_buffer, inp_copy);
    
    int result_dims[1] = {1};
    DoubleTensor result = double_tensor_alloc(result_dims, 1);
    
    double loss = 0.0;
    double eps = 1e-7;
    
    if (inp.num_elem == target.num_elem) {
        for (int i = 0; i < inp.num_elem; i++) {
            double pred = inp.elems[i];
            pred = pred < eps ? eps : (pred > 1.0 - eps ? 1.0 - eps : pred);
            loss -= target.elems[i] * log(pred);
        }
        loss /= inp.num_elem;
    } else {
        for (int i = 0; i < target.num_elem; i++) {
            int class_idx = (int)target.elems[i];
            if (class_idx >= 0 && class_idx < inp.num_elem) {
                double pred = inp.elems[class_idx];
                pred = pred < eps ? eps : (pred > 1.0 - eps ? 1.0 - eps : pred);
                loss -= log(pred);
            }
        }
        loss /= target.num_elem;
    }
    
    result.elems[0] = loss;
    return result;
}

DoubleTensor cross_entropy_backward(CrossEntropy* cross_entropy, DoubleTensor preds) {
    if (cross_entropy->inp_buffer.inp_buffer_size == 0) {
        DoubleTensor empty = {0};
        return empty;
    }
    
    DoubleTensor inp_grad = double_tensor_alloc(preds.dims, preds.num_dims);
    
    for (int i = 0; i < preds.num_elem; i++) {
        inp_grad.elems[i] = preds.elems[i];
    }
    
    tensor_buffer_add(&cross_entropy->inp_grad_buffer, inp_grad);
    
    return inp_grad;
}

void cross_entropy_free(CrossEntropy* cross_entropy) {
    tensor_buffer_free(&cross_entropy->inp_buffer);
    tensor_buffer_free(&cross_entropy->inp_grad_buffer);
    
    if (cross_entropy->name) {
        free(cross_entropy->name);
        cross_entropy->name = NULL;
    }
}