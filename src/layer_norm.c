#include "../include/layer_norm.h"
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

LayerNorm layer_norm_create(int features) {
    LayerNorm layer_norm;
    layer_norm.features = features;
    
    tensor_buffer_init(&layer_norm.inp_buffer);
    tensor_buffer_init(&layer_norm.inp_grad_buffer);
    
    layer_norm.name = (char*)malloc(11 * sizeof(char));
    strcpy(layer_norm.name, "layer_norm");
    
    return layer_norm;
}

DoubleTensor layer_norm_forward(LayerNorm* layer_norm, DoubleTensor inp) {
    DoubleTensor inp_copy;
    inp_copy.dims = (int*)malloc(inp.num_dims * sizeof(int));
    memcpy(inp_copy.dims, inp.dims, inp.num_dims * sizeof(int));
    inp_copy.num_dims = inp.num_dims;
    inp_copy.num_elem = inp.num_elem;
    inp_copy.elems = (double*)malloc(inp.num_elem * sizeof(double));
    memcpy(inp_copy.elems, inp.elems, inp.num_elem * sizeof(double));
    
    tensor_buffer_add(&layer_norm->inp_buffer, inp_copy);
    
    DoubleTensor result = double_tensor_alloc(inp.dims, inp.num_dims);
    
    int batch_size = inp.num_elem / layer_norm->features;
    
    for (int b = 0; b < batch_size; b++) {
        double mean = 0.0;
        for (int i = 0; i < layer_norm->features; i++) {
            mean += inp.elems[b * layer_norm->features + i];
        }
        mean /= layer_norm->features;
        
        double variance = 0.0;
        for (int i = 0; i < layer_norm->features; i++) {
            double diff = inp.elems[b * layer_norm->features + i] - mean;
            variance += diff * diff;
        }
        variance /= layer_norm->features;
        
        double std_dev = sqrt(variance + 1e-5);
        
        for (int i = 0; i < layer_norm->features; i++) {
            result.elems[b * layer_norm->features + i] = 
                (inp.elems[b * layer_norm->features + i] - mean) / std_dev;
        }
    }
    
    return result;
}

DoubleTensor layer_norm_backward(LayerNorm* layer_norm, DoubleTensor delta) {
    if (layer_norm->inp_buffer.inp_buffer_size == 0) {
        DoubleTensor empty = {0};
        return empty;
    }
    
    DoubleTensor inp_grad = double_tensor_alloc(delta.dims, delta.num_dims);
    
    for (int i = 0; i < delta.num_elem; i++) {
        inp_grad.elems[i] = delta.elems[i];
    }
    
    tensor_buffer_add(&layer_norm->inp_grad_buffer, inp_grad);
    
    return inp_grad;
}

void layer_norm_free(LayerNorm* layer_norm) {
    tensor_buffer_free(&layer_norm->inp_buffer);
    tensor_buffer_free(&layer_norm->inp_grad_buffer);
    
    if (layer_norm->name) {
        free(layer_norm->name);
        layer_norm->name = NULL;
    }
}