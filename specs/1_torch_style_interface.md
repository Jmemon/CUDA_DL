

Language:
 - C



==== tensor.c ====
struct DoubleTensor
 - elems: double* (pointer to values)
 - dims: int* (ie how to stride through the values)
 - num_dims: int (number of dimensions in the tensor)
 - num_elem: int (number of elements in the tensor)

methods
double_tensor_alloc(dims: int[]) -> DoubleTensor
double_tensor_random_init(tensor: *DoubleTensor) -> void
double_tensor_zero_init(tensor: *DoubleTensor) -> void
double_tensor_set(tensor: *DoubleTensor, index: int[], value: double) -> void
double_tensor_add(tensor1: *DoubleTensor, tensor2: *DoubleTensor) -> DoubleTensor // elementwise add, check dims and types
double_tensor_matmul(tensor1: *DoubleTensor, tensor2: *DoubleTensor) -> DoubleTensor // matmul, check dims and types
double_tensor_free(tensor: *DoubleTensor) -> void // deallocates all tensors in the tensor


==== resnet18.c ====
struct ResidualBlock:
- conv1: Conv2d
- conv2: Conv2d
- layer_norm1: LayerNorm
- layer_norm2: LayerNorm
- relu1: Relu
- relu2: Relu
- name: string = "residual_block"

methods
residual_block_create(channels: int, kernel_size: int, stride: int = 1, padding: int = 0) -> ResidualBlock  // in_channels==out_channels for skip connection, kernel_size==3 by default
residual_block_forward(residual_block: *ResidualBlock, x: DoubleTensor) -> DoubleTensor
residual_block_backward(residual_block: *ResidualBlock, delta: DoubleTensor) -> DoubleTensor
residual_block_destroy(residual_block: *ResidualBlock) -> void // deallocates all tensors in the residual block


struct ResNet18:
- conv1: Conv2d  // 64 channels, 7x7 kernel, stride 2
- layer_norm1: LayerNorm
- relu1: Relu
- max_pool1: MaxPool2d
- residual_block1: ResidualBlock  // 64 channels, 3x3 kernel, stride 1
- residual_block2: ResidualBlock  // 64 channels, 3x3 kernel, stride 1
- residual_block3: ResidualBlock  // 128 channels, 3x3 kernel, stride 2
- residual_block4: ResidualBlock  // 128 channels, 3x3 kernel, stride 1
- global_avg_pool: GlobalAvgPool2d  // avg over channels, so will have 128dim output
- linear: Linear  // 128 -> num_classes
- name: string = "resnet18"

methods
resnet18_create(num_classes: int) -> ResNet18
resnet18_forward(resnet18: *ResNet18, x: DoubleTensor) -> DoubleTensor
resnet18_backward(resnet18: *ResNet18, delta: DoubleTensor) -> DoubleTensor
resnet18_destroy(resnet18: *ResNet18) -> void // deallocates all tensors in the resnet18, calls destroy on all submodules


==== linear.c ====
struct Linear:
 - features_in: int
 - features_out: int
 - weights: DoubleTensor
 - bias: DoubleTensor
 - inp_buffer: vector<DoubleTensor*>  // for backprop
 - weights_grad_buffer: vector<DoubleTensor*>  // for backprop
 - bias_grad_buffer: vector<DoubleTensor*>  // for backprop
 - inp_grad_buffer: vector<DoubleTensor*>  // for backprop
 - name: string = "linear"

methods
linear_create(features_in: int, features_out: int) -> Linear
linear_forward(linear: *Linear, inp: DoubleTensor) -> DoubleTensor
linear_backward(linear: *Linear, delta: DoubleTensor) -> DoubleTensor
linear_destroy(linear: *Linear) -> void // deallocates all tensors in the linear


==== conv2d.c ====
struct Conv2d:
 - in_channels: int
 - out_channels: int
 - kernel_size: int
 - stride: int
 - padding: int
 - kernel: DoubleTensor
 - inp_buffer: vector<DoubleTensor*>  // for backprop
 - kernel_grad_buffer: vector<DoubleTensor*>  // for backprop
 - inp_grad_buffer: vector<DoubleTensor*>  // for backprop
 - name: string = "conv2d"

methods
conv2d_create(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0) -> Conv2d
conv2d_forward(conv2d: *Conv2d, inp: DoubleTensor) -> DoubleTensor
conv2d_backward(conv2d: *Conv2d, delta: DoubleTensor) -> DoubleTensor
conv2d_destroy(conv2d: *Conv2d) -> void // deallocates all tensors in the conv2d


==== relu.c ====
struct Relu:
 - inp_buffer: vector<DoubleTensor*>  // for backprop
 - inp_grad_buffer: vector<DoubleTensor*>  // for backprop
 - name: string = "relu"

methods
relu_create() -> Relu
relu_forward(relu: *Relu, inp: DoubleTensor) -> DoubleTensor
relu_backward(relu: *Relu, delta: DoubleTensor) -> DoubleTensor
relu_destroy(relu: *Relu) -> void // deallocates all tensors in the relu


==== softmax.c ====
struct Softmax:
 - inp_buffer: vector<DoubleTensor*>  // for backprop
 - inp_grad_buffer: vector<DoubleTensor*>  // for backprop
 - name: string = "softmax"

methods
softmax_create() -> Softmax
softmax_forward(softmax: *Softmax, inp: DoubleTensor) -> DoubleTensor
softmax_backward(softmax: *Softmax, delta: DoubleTensor) -> DoubleTensor
softmax_destroy(softmax: *Softmax) -> void // deallocates all tensors in the softmax


==== max_pool2d.c ====
struct MaxPool2d:
 - kernel_size: int
 - inp_buffer: vector<DoubleTensor*>  // for backprop
 - inp_grad_buffer: vector<DoubleTensor*>  // for backprop
 - name: string = "max_pool2d"

methods
max_pool2d_create(kernel_size: int) -> MaxPool2d
max_pool2d_forward(max_pool2d: *MaxPool2d, inp: DoubleTensor) -> DoubleTensor
max_pool2d_backward(max_pool2d: *MaxPool2d, delta: DoubleTensor) -> DoubleTensor
max_pool2d_destroy(max_pool2d: *MaxPool2d) -> void // deallocates all tensors in the max_pool2d


==== global_avg_pool2d.c ====
struct GlobalAvgPool2d:
 - inp_buffer: vector<DoubleTensor*>  // for backprop
 - inp_grad_buffer: vector<DoubleTensor*>  // for backprop
 - name: string = "global_avg_pool2d"

methods
global_avg_pool2d_create() -> GlobalAvgPool2d
global_avg_pool2d_forward(global_avg_pool2d: *GlobalAvgPool2d, inp: DoubleTensor) -> DoubleTensor
global_avg_pool2d_backward(global_avg_pool2d: *GlobalAvgPool2d, delta: DoubleTensor) -> DoubleTensor
global_avg_pool2d_destroy(global_avg_pool2d: *GlobalAvgPool2d) -> void // deallocates all tensors in the global_avg_pool2d


==== layer_norm.c ====
struct LayerNorm:
 - features: int
 - inp_buffer: vector<DoubleTensor*>  // for backprop
 - inp_grad_buffer: vector<DoubleTensor*>  // for backprop
 - name: string = "layer_norm"

methods
layer_norm_create(features: int) -> LayerNorm
layer_norm_forward(layer_norm: *LayerNorm, inp: DoubleTensor) -> DoubleTensor
layer_norm_backward(layer_norm: *LayerNorm, delta: DoubleTensor) -> DoubleTensor
layer_norm_destroy(layer_norm: *LayerNorm) -> void // deallocates all tensors in the layer_norm


==== adam.c ====
struct Adam:
 - lr: float
 - beta1: float
 - beta2: float
 - eps: float
 - name: string = "adam"

methods
adam_create(lr: float, beta1: float, beta2: float, eps: float) -> Adam
adam_step(adam: *Adam, module: *Module) -> void
adam_destroy(adam: *Adam) -> void // deallocates all tensors in the adam


==== cross_entropy.c ====
struct CrossEntropy:
 - inp_buffer: vector<DoubleTensor*>  // for backprop
 - inp_grad_buffer: vector<DoubleTensor*>  // for backprop
 - name: string = "cross_entropy"

methods
cross_entropy_create() -> CrossEntropy
cross_entropy_forward(cross_entropy: *CrossEntropy, inp: DoubleTensor, target: DoubleTensor) -> DoubleTensor
cross_entropy_backward(cross_entropy: *CrossEntropy, preds: DoubleTensor) -> DoubleTensor
cross_entropy_destroy(cross_entropy: *CrossEntropy) -> void // deallocates all tensors in the cross_entropy


==== mse.c ====
struct MSE:
 - inp_buffer: vector<DoubleTensor*>  // for backprop
 - inp_grad_buffer: vector<DoubleTensor*>  // for backprop
 - name: string = "mse"

methods
mse_create() -> MSE
mse_forward(mse: *MSE, inp: DoubleTensor, target: DoubleTensor) -> DoubleTensor
mse_backward(mse: *MSE, preds: DoubleTensor) -> DoubleTensor
mse_destroy(mse: *MSE) -> void // deallocates all tensors in the mse


==== typecheck.c ====
valid_layers: string[4] = ["conv2d", "linear", "max_pool2d", "layer_norm", "global_avg_pool2d"]
valid_activations: string[2] = ["relu", "softmax"]
valid_losses: string[2] = ["cross_entropy", "mse"]
valid_optimizers: string[1] = ["adam"]


===== Final interface: Cifar10 example Resnet – that this runs signifies the interface is correct and we can move on to cuda kernels =====

int main() {
    resnet = resnet18_create(...)
    loss = cross_entropy_create(...)
    optimizer = adam_create(...)
    cifar10 = cifar10_load(...)

    for (int i = 0; i < 100; i++) {
        ...
    }
}

============================================
