#include "../include/resnet18.h"
#include <stdlib.h>
#include <string.h>

ResidualBlock residual_block_create(int channels, int kernel_size, int stride, int padding) {
    ResidualBlock block;
    
    block.conv1 = conv2d_create(channels, channels, kernel_size, stride, padding);
    block.conv2 = conv2d_create(channels, channels, kernel_size, 1, padding);
    
    block.layer_norm1 = layer_norm_create(channels);
    block.layer_norm2 = layer_norm_create(channels);
    
    block.relu1 = relu_create();
    block.relu2 = relu_create();
    
    block.name = (char*)malloc(16 * sizeof(char));
    strcpy(block.name, "residual_block");
    
    return block;
}

DoubleTensor residual_block_forward(ResidualBlock* residual_block, DoubleTensor x) {
    DoubleTensor identity = x;
    
    DoubleTensor out = conv2d_forward(&residual_block->conv1, x);
    if (out.elems == NULL) {
        return out;
    }
    
    out = relu_forward(&residual_block->relu1, out);
    
    DoubleTensor out2 = conv2d_forward(&residual_block->conv2, out);
    double_tensor_free(&out);
    if (out2.elems == NULL) {
        return out2;
    }
    
    DoubleTensor result = double_tensor_add(&out2, &identity);
    double_tensor_free(&out2);
    
    if (result.elems == NULL) {
        return result;
    }
    
    DoubleTensor final_result = relu_forward(&residual_block->relu2, result);
    double_tensor_free(&result);
    
    return final_result;
}

DoubleTensor residual_block_backward(ResidualBlock* residual_block, DoubleTensor delta) {
    DoubleTensor grad = relu_backward(&residual_block->relu2, delta);
    
    DoubleTensor conv2_grad = conv2d_backward(&residual_block->conv2, grad);
    DoubleTensor relu1_grad = relu_backward(&residual_block->relu1, conv2_grad);
    DoubleTensor conv1_grad = conv2d_backward(&residual_block->conv1, relu1_grad);
    
    DoubleTensor skip_grad = grad;
    
    DoubleTensor result = double_tensor_add(&conv1_grad, &skip_grad);
    
    double_tensor_free(&conv2_grad);
    double_tensor_free(&relu1_grad);
    double_tensor_free(&conv1_grad);
    
    return result;
}

void residual_block_free(ResidualBlock* residual_block) {
    conv2d_free(&residual_block->conv1);
    conv2d_free(&residual_block->conv2);
    layer_norm_free(&residual_block->layer_norm1);
    layer_norm_free(&residual_block->layer_norm2);
    relu_free(&residual_block->relu1);
    relu_free(&residual_block->relu2);
    
    if (residual_block->name) {
        free(residual_block->name);
        residual_block->name = NULL;
    }
}

ResNet18 resnet18_create(int num_classes) {
    ResNet18 resnet;
    
    resnet.conv1 = conv2d_create(3, 64, 7, 2, 3);
    resnet.layer_norm1 = layer_norm_create(64);
    resnet.relu1 = relu_create();
    resnet.max_pool1 = max_pool2d_create(3);
    
    resnet.residual_block1 = residual_block_create(64, 3, 1, 1);
    resnet.residual_block2 = residual_block_create(64, 3, 1, 1);
    resnet.residual_block3 = residual_block_create(128, 3, 2, 1);
    resnet.residual_block4 = residual_block_create(128, 3, 1, 1);
    
    resnet.global_avg_pool = global_avg_pool2d_create();
    resnet.linear = linear_create(128, num_classes);
    
    resnet.name = (char*)malloc(8 * sizeof(char));
    strcpy(resnet.name, "resnet18");
    
    return resnet;
}

DoubleTensor resnet18_forward(ResNet18* resnet18, DoubleTensor x) {
    DoubleTensor out = conv2d_forward(&resnet18->conv1, x);
    if (out.elems == NULL) return out;
    
    DoubleTensor out2 = relu_forward(&resnet18->relu1, out);
    double_tensor_free(&out);
    
    DoubleTensor out3 = max_pool2d_forward(&resnet18->max_pool1, out2);
    double_tensor_free(&out2);
    if (out3.elems == NULL) return out3;
    
    DoubleTensor out4 = residual_block_forward(&resnet18->residual_block1, out3);
    double_tensor_free(&out3);
    if (out4.elems == NULL) return out4;
    
    DoubleTensor out5 = residual_block_forward(&resnet18->residual_block2, out4);
    double_tensor_free(&out4);
    if (out5.elems == NULL) return out5;
    
    DoubleTensor out6 = residual_block_forward(&resnet18->residual_block3, out5);
    double_tensor_free(&out5);
    if (out6.elems == NULL) return out6;
    
    DoubleTensor out7 = residual_block_forward(&resnet18->residual_block4, out6);
    double_tensor_free(&out6);
    if (out7.elems == NULL) return out7;
    
    DoubleTensor out8 = global_avg_pool2d_forward(&resnet18->global_avg_pool, out7);
    double_tensor_free(&out7);
    if (out8.elems == NULL) return out8;
    
    DoubleTensor result = linear_forward(&resnet18->linear, out8);
    double_tensor_free(&out8);
    
    return result;
}

DoubleTensor resnet18_backward(ResNet18* resnet18, DoubleTensor delta) {
    DoubleTensor grad = linear_backward(&resnet18->linear, delta);
    
    grad = global_avg_pool2d_backward(&resnet18->global_avg_pool, grad);
    grad = residual_block_backward(&resnet18->residual_block4, grad);
    grad = residual_block_backward(&resnet18->residual_block3, grad);
    grad = residual_block_backward(&resnet18->residual_block2, grad);
    grad = residual_block_backward(&resnet18->residual_block1, grad);
    grad = max_pool2d_backward(&resnet18->max_pool1, grad);
    grad = relu_backward(&resnet18->relu1, grad);
    grad = conv2d_backward(&resnet18->conv1, grad);
    
    return grad;
}

void resnet18_free(ResNet18* resnet18) {
    conv2d_free(&resnet18->conv1);
    layer_norm_free(&resnet18->layer_norm1);
    relu_free(&resnet18->relu1);
    max_pool2d_free(&resnet18->max_pool1);
    
    residual_block_free(&resnet18->residual_block1);
    residual_block_free(&resnet18->residual_block2);
    residual_block_free(&resnet18->residual_block3);
    residual_block_free(&resnet18->residual_block4);
    
    global_avg_pool2d_free(&resnet18->global_avg_pool);
    linear_free(&resnet18->linear);
    
    if (resnet18->name) {
        free(resnet18->name);
        resnet18->name = NULL;
    }
}