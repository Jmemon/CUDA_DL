
#### Module class with forward and backward methods. Modules are composable.
std::vector<double> parameters; 
// parameters of the module

std::vector<double> input_buffer;  
// buffer for inputs (to be used in backward pass)

std::vector<double> parameter_gradients;
// buffer for gradients of loss wrt parameters
// this would be changed if we implemented automatic differentiation, gradients would be computed on the fly and associated with each vector.

std::vector<double> input_gradients;
// buffer for gradients of loss wrt inputs
// this would be changed if we implemented automatic differentiation, gradients would be computed on the fly and associated with each vector.

virtual std::vector<double> forward(std::vector<double> &x); 
// forward pass

virtual std::vector<double> backward(std::vector<double> &dL_dz); 
// backward pass 
// dL_dz is the gradient of the loss with respect to the output of the module
// for gradients wrt to parameters, we need the input to the module (stored in parameters)
// for gradients wrt to inputs, we need the module parameters (stored in in_buffer)


#### Implement Linear, Conv2D, LayerNorm, ReLU, Sigmoid, LeakyReLU, Tanh, CrossEntropyLoss, MSELoss.
Linear : Module (in_size, out_size): y = Wx + b
Conv2D : Module (in_channels, out_channels, kernel_size, stride, padding):
LayerNorm : Module (): y = (x - mean(x)) / sqrt(var(x) + eps)
ReLU : Module (): y = max(0, x)
Sigmoid : Module (): y = 1 / (1 + exp(-x))
LeakyReLU : Module (): y = max(0.01x, x)
Tanh : Module (): y = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
CrossEntropyLoss : Module (num_classes): y = -sum(y_i * log(p_i))
MSELoss : Module (): y = sum((y_i - p_i)^2)


#### Optimizer class with step method.
virtual void step()


#### Implement Adam.
Adam : Optimizer (network, learning_rate, beta1, beta2, epsilon)


Cifar10 classification as an example.
