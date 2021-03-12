#ifndef NEURALNET_H
#define NEURALNET_H

#include "Activation.cuh"
#include "Loss.cuh"
#include <vector>

// more detailed information on each function is in src/NeuralNet.cu

typedef enum Activation {
	binary_step,
	sigmoid,
	// tanh,
	relu,
	leaky_relu,
	exponential
} Activation;

typedef enum Loss {
	mse,
	logLoss
} Loss;


#endif	// NEURALNET_H
