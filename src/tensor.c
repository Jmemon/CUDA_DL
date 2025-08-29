#include "../include/tensor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

DoubleTensor double_tensor_alloc(int* dims, int num_dims) {
    DoubleTensor tensor;
    tensor.num_dims = num_dims;
    tensor.dims = (int*)malloc(num_dims * sizeof(int));
    
    int total_elements = 1;
    for (int i = 0; i < num_dims; i++) {
        tensor.dims[i] = dims[i];
        total_elements *= dims[i];
    }
    
    tensor.num_elem = total_elements;
    tensor.elems = (double*)malloc(total_elements * sizeof(double));
    
    return tensor;
}

void double_tensor_random_init(DoubleTensor* tensor) {
    for (int i = 0; i < tensor->num_elem; i++) {
        tensor->elems[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }
}

void double_tensor_randn_init(DoubleTensor* tensor) {
    for (int i = 0; i < tensor->num_elem; i += 2) {
        static int has_spare = 0;
        static double spare;
        
        if (has_spare) {
            tensor->elems[i] = spare;
            has_spare = 0;
        } else {
            double u = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
            double v = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
            double s = u * u + v * v;
            
            while (s >= 1.0 || s == 0.0) {
                u = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
                v = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
                s = u * u + v * v;
            }
            
            double factor = sqrt(-2.0 * log(s) / s);
            tensor->elems[i] = u * factor;
            
            if (i + 1 < tensor->num_elem) {
                tensor->elems[i + 1] = v * factor;
            } else {
                spare = v * factor;
                has_spare = 1;
            }
        }
    }
}

void double_tensor_he_init(DoubleTensor* tensor, int fan_in) {
    double std_dev = sqrt(2.0 / fan_in);
    double_tensor_randn_init(tensor);
    
    for (int i = 0; i < tensor->num_elem; i++) {
        tensor->elems[i] *= std_dev;
    }
}

void double_tensor_zero_init(DoubleTensor* tensor) {
    memset(tensor->elems, 0, tensor->num_elem * sizeof(double));
}

void double_tensor_set(DoubleTensor* tensor, int* index, double value) {
    int flat_index = 0;
    int stride = 1;
    
    for (int i = tensor->num_dims - 1; i >= 0; i--) {
        flat_index += index[i] * stride;
        stride *= tensor->dims[i];
    }
    
    tensor->elems[flat_index] = value;
}

DoubleTensor double_tensor_add(DoubleTensor* tensor1, DoubleTensor* tensor2) {
    if (tensor1->num_dims != tensor2->num_dims || tensor1->num_elem != tensor2->num_elem) {
        DoubleTensor empty = {0};
        return empty;
    }
    
    for (int i = 0; i < tensor1->num_dims; i++) {
        if (tensor1->dims[i] != tensor2->dims[i]) {
            DoubleTensor empty = {0};
            return empty;
        }
    }
    
    DoubleTensor result = double_tensor_alloc(tensor1->dims, tensor1->num_dims);
    for (int i = 0; i < result.num_elem; i++) {
        result.elems[i] = tensor1->elems[i] + tensor2->elems[i];
    }
    
    return result;
}

DoubleTensor double_tensor_matmul(DoubleTensor* tensor1, DoubleTensor* tensor2) {
    if (tensor1->num_dims != 2 || tensor2->num_dims != 2) {
        DoubleTensor empty = {0};
        return empty;
    }
    
    int m = tensor1->dims[0];
    int k = tensor1->dims[1];
    int n = tensor2->dims[1];
    
    if (k != tensor2->dims[0]) {
        DoubleTensor empty = {0};
        return empty;
    }
    
    int result_dims[2] = {m, n};
    DoubleTensor result = double_tensor_alloc(result_dims, 2);
    double_tensor_zero_init(&result);
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int l = 0; l < k; l++) {
                result.elems[i * n + j] += tensor1->elems[i * k + l] * tensor2->elems[l * n + j];
            }
        }
    }
    
    return result;
}

void double_tensor_free(DoubleTensor* tensor) {
    if (tensor->elems) {
        free(tensor->elems);
        tensor->elems = NULL;
    }
    if (tensor->dims) {
        free(tensor->dims);
        tensor->dims = NULL;
    }
    tensor->num_dims = 0;
    tensor->num_elem = 0;
}