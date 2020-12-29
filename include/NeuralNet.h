#ifndef NEURALNET_H
#define NEURALNET_H

#include "Activation.cuh"
#include "Loss.cuh"
#include <vector>

typedef enum Activation {
	binary_step,
	sigmoid,
	// tanh,
	relu,
	leaky_relu
} Activation;

typedef enum Loss {
	mse,
	logLoss
} Loss;

class NeuralNet {
	private:
		const std::vector<int> layers; 
		// position i containts num nodes in layer i (first layer is layer 0)

		std::vector<std::vector<double> > weights; 
		// vector containing weight matrixes in row major form (vect of vects)

		const std::vector<Activation> funcs; 
		// vector containing activation func to apply to (W_i)x_i at position i

		const Loss errFunc;
		// stores the error func to use 

	public:
		NeuralNet(std::vector<int> &l, std::vector<Activation> &f, Loss e); 
		// contructor which initializes data members
		// takes vectors of layer sizes and activation functions as input

		std::vector<double> activation(std::vector<double> &x, Activation f);
		// applies activation function to x
		// returns output in vector
	
		std::vector<double> forwardPass(std::vector<double> &x);
		// x can be a matrix in row-major form
		// We use layers[0] to determine the batchsize
		// returns vector<double>
		// 	this vector contains the z's for each layer and aL (output)
		// 	use the layers vector to determine where each starts and ends
	
		double calcLoss(std::vector<double> &x, std::vector<double> &y);
		// x can be matrix in row-major form
		// use layer[layers.size() - 1] to detetmine output size
		// returns double which is average of error for each sample in batch
		// 	SO BE CAREFUL ABOUT ONLY PUTTING ONE BATCH IN AT A TIME
	
		void printNN() const;
		// prints info about the neural net
		// Layer Sizes
		// Activation Funcs at each layer
		// Weight Matrix Dims
		// Error Function

		void printWeights(int l) const;		
		// prints the weights
		// l specifies the layer (0 to num_layers - 2)

};

#endif	// NEURALNET_H
