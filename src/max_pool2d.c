#include "../include/max_pool2d.h"
#include <stdlib.h>
#include <string.h>
#include <float.h>

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

MaxPool2d max_pool2d_create(int kernel_size) {
    MaxPool2d max_pool2d;
    max_pool2d.kernel_size = kernel_size;
    
    tensor_buffer_init(&max_pool2d.inp_buffer);
    tensor_buffer_init(&max_pool2d.inp_grad_buffer);
    
    max_pool2d.name = (char*)malloc(12 * sizeof(char));
    strcpy(max_pool2d.name, "max_pool2d");
    
    return max_pool2d;
}

DoubleTensor max_pool2d_forward(MaxPool2d* max_pool2d, DoubleTensor inp) {
    DoubleTensor inp_copy;
    inp_copy.dims = (int*)malloc(inp.num_dims * sizeof(int));
    memcpy(inp_copy.dims, inp.dims, inp.num_dims * sizeof(int));
    inp_copy.num_dims = inp.num_dims;
    inp_copy.num_elem = inp.num_elem;
    inp_copy.elems = (double*)malloc(inp.num_elem * sizeof(double));
    memcpy(inp_copy.elems, inp.elems, inp.num_elem * sizeof(double));
    
    tensor_buffer_add(&max_pool2d->inp_buffer, inp_copy);
    
    if (inp.num_dims != 3 && inp.num_dims != 4) {
        DoubleTensor empty = {0};
        return empty;
    }
    
    int batch_size = (inp.num_dims == 4) ? inp.dims[0] : 1;
    int channels = (inp.num_dims == 4) ? inp.dims[1] : inp.dims[0];
    int in_height = (inp.num_dims == 4) ? inp.dims[2] : inp.dims[1];
    int in_width = (inp.num_dims == 4) ? inp.dims[3] : inp.dims[2];
    
    int out_height = in_height / max_pool2d->kernel_size;
    int out_width = in_width / max_pool2d->kernel_size;
    
    int result_dims[4] = {batch_size, channels, out_height, out_width};
    DoubleTensor result = double_tensor_alloc(result_dims, inp.num_dims);
    
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < channels; c++) {
            for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                    double max_val = -DBL_MAX;
                    
                    for (int kh = 0; kh < max_pool2d->kernel_size; kh++) {
                        for (int kw = 0; kw < max_pool2d->kernel_size; kw++) {
                            int ih = oh * max_pool2d->kernel_size + kh;
                            int iw = ow * max_pool2d->kernel_size + kw;
                            
                            int inp_idx;
                            if (inp.num_dims == 4) {
                                inp_idx = b * channels * in_height * in_width + 
                                         c * in_height * in_width + ih * in_width + iw;
                            } else {
                                inp_idx = c * in_height * in_width + ih * in_width + iw;
                            }
                            
                            if (inp.elems[inp_idx] > max_val) {
                                max_val = inp.elems[inp_idx];
                            }
                        }
                    }
                    
                    int result_idx;
                    if (result.num_dims == 4) {
                        result_idx = b * channels * out_height * out_width + 
                                    c * out_height * out_width + oh * out_width + ow;
                    } else {
                        result_idx = c * out_height * out_width + oh * out_width + ow;
                    }
                    
                    result.elems[result_idx] = max_val;
                }
            }
        }
    }
    
    return result;
}

DoubleTensor max_pool2d_backward(MaxPool2d* max_pool2d, DoubleTensor delta) {
    if (max_pool2d->inp_buffer.inp_buffer_size == 0) {
        DoubleTensor empty = {0};
        return empty;
    }
    
    DoubleTensor last_inp = max_pool2d->inp_buffer.inp_buffer[max_pool2d->inp_buffer.inp_buffer_size - 1];
    
    DoubleTensor inp_grad = double_tensor_alloc(last_inp.dims, last_inp.num_dims);
    double_tensor_zero_init(&inp_grad);
    
    tensor_buffer_add(&max_pool2d->inp_grad_buffer, inp_grad);
    
    return inp_grad;
}

void max_pool2d_free(MaxPool2d* max_pool2d) {
    tensor_buffer_free(&max_pool2d->inp_buffer);
    tensor_buffer_free(&max_pool2d->inp_grad_buffer);
    
    if (max_pool2d->name) {
        free(max_pool2d->name);
        max_pool2d->name = NULL;
    }
}