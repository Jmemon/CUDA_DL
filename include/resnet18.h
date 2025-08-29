#ifndef RESNET18_H
#define RESNET18_H

#include "tensor.h"
#include "conv2d.h"
#include "linear.h"
#include "relu.h"
#include "layer_norm.h"
#include "max_pool2d.h"
#include "global_avg_pool2d.h"

typedef struct {
    Conv2d conv1;
    Conv2d conv2;
    LayerNorm layer_norm1;
    LayerNorm layer_norm2;
    Relu relu1;
    Relu relu2;
    char* name;
} ResidualBlock;

typedef struct {
    Conv2d conv1;
    LayerNorm layer_norm1;
    Relu relu1;
    MaxPool2d max_pool1;
    ResidualBlock residual_block1;
    ResidualBlock residual_block2;
    ResidualBlock residual_block3;
    ResidualBlock residual_block4;
    GlobalAvgPool2d global_avg_pool;
    Linear linear;
    char* name;
} ResNet18;

ResidualBlock residual_block_create(int channels, int kernel_size, int stride, int padding);
DoubleTensor residual_block_forward(ResidualBlock* residual_block, DoubleTensor x);
DoubleTensor residual_block_backward(ResidualBlock* residual_block, DoubleTensor delta);
void residual_block_free(ResidualBlock* residual_block);

ResNet18 resnet18_create(int num_classes);
DoubleTensor resnet18_forward(ResNet18* resnet18, DoubleTensor x);
DoubleTensor resnet18_backward(ResNet18* resnet18, DoubleTensor delta);
void resnet18_free(ResNet18* resnet18);

#endif