#include "../include/global_avg_pool2d.h"
#include <stdlib.h>
#include <string.h>

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

GlobalAvgPool2d global_avg_pool2d_create(void) {
    GlobalAvgPool2d global_avg_pool2d;
    
    tensor_buffer_init(&global_avg_pool2d.inp_buffer);
    tensor_buffer_init(&global_avg_pool2d.inp_grad_buffer);
    
    global_avg_pool2d.name = (char*)malloc(19 * sizeof(char));
    strcpy(global_avg_pool2d.name, "global_avg_pool2d");
    
    return global_avg_pool2d;
}

DoubleTensor global_avg_pool2d_forward(GlobalAvgPool2d* global_avg_pool2d, DoubleTensor inp) {
    DoubleTensor inp_copy;
    inp_copy.dims = (int*)malloc(inp.num_dims * sizeof(int));
    memcpy(inp_copy.dims, inp.dims, inp.num_dims * sizeof(int));
    inp_copy.num_dims = inp.num_dims;
    inp_copy.num_elem = inp.num_elem;
    inp_copy.elems = (double*)malloc(inp.num_elem * sizeof(double));
    memcpy(inp_copy.elems, inp.elems, inp.num_elem * sizeof(double));
    
    tensor_buffer_add(&global_avg_pool2d->inp_buffer, inp_copy);
    
    if (inp.num_dims != 3 && inp.num_dims != 4) {
        DoubleTensor empty = {0};
        return empty;
    }
    
    int batch_size = (inp.num_dims == 4) ? inp.dims[0] : 1;
    int channels = (inp.num_dims == 4) ? inp.dims[1] : inp.dims[0];
    int height = (inp.num_dims == 4) ? inp.dims[2] : inp.dims[1];
    int width = (inp.num_dims == 4) ? inp.dims[3] : inp.dims[2];
    
    int spatial_size = height * width;
    
    int result_dims[2] = {batch_size, channels};
    DoubleTensor result = double_tensor_alloc(result_dims, inp.num_dims == 4 ? 2 : 1);
    double_tensor_zero_init(&result);
    
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < channels; c++) {
            double sum = 0.0;
            
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int inp_idx;
                    if (inp.num_dims == 4) {
                        inp_idx = b * channels * height * width + c * height * width + h * width + w;
                    } else {
                        inp_idx = c * height * width + h * width + w;
                    }
                    sum += inp.elems[inp_idx];
                }
            }
            
            int result_idx = (result.num_dims == 2) ? b * channels + c : c;
            result.elems[result_idx] = sum / spatial_size;
        }
    }
    
    return result;
}

DoubleTensor global_avg_pool2d_backward(GlobalAvgPool2d* global_avg_pool2d, DoubleTensor delta) {
    if (global_avg_pool2d->inp_buffer.inp_buffer_size == 0) {
        DoubleTensor empty = {0};
        return empty;
    }
    
    DoubleTensor last_inp = global_avg_pool2d->inp_buffer.inp_buffer[global_avg_pool2d->inp_buffer.inp_buffer_size - 1];
    
    DoubleTensor inp_grad = double_tensor_alloc(last_inp.dims, last_inp.num_dims);
    double_tensor_zero_init(&inp_grad);
    
    tensor_buffer_add(&global_avg_pool2d->inp_grad_buffer, inp_grad);
    
    return inp_grad;
}

void global_avg_pool2d_free(GlobalAvgPool2d* global_avg_pool2d) {
    tensor_buffer_free(&global_avg_pool2d->inp_buffer);
    tensor_buffer_free(&global_avg_pool2d->inp_grad_buffer);
    
    if (global_avg_pool2d->name) {
        free(global_avg_pool2d->name);
        global_avg_pool2d->name = NULL;
    }
}