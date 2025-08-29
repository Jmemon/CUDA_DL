#ifndef TENSOR_H
#define TENSOR_H

typedef struct {
    double* elems;
    int* dims;
    int num_dims;
    int num_elem;
} DoubleTensor;

DoubleTensor double_tensor_alloc(int* dims, int num_dims);
void double_tensor_random_init(DoubleTensor* tensor);
void double_tensor_randn_init(DoubleTensor* tensor);
void double_tensor_he_init(DoubleTensor* tensor, int fan_in);
void double_tensor_zero_init(DoubleTensor* tensor);
void double_tensor_set(DoubleTensor* tensor, int* index, double value);
DoubleTensor double_tensor_add(DoubleTensor* tensor1, DoubleTensor* tensor2);
DoubleTensor double_tensor_matmul(DoubleTensor* tensor1, DoubleTensor* tensor2);
void double_tensor_free(DoubleTensor* tensor);

#endif