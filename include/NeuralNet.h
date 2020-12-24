#ifndef NEURALNET_H
#define NEURALNET_H

#include "Activation.cuh"
#include <vector>

typedef enum Activation {
	binary_step,
	sigmoid,
	// tanh,
	relu,
	leaky_relu
} Activation;

class NeuralNet {
	private:
		const int num_layers;
		// number of layers in neural net

		const std::vector<int> layers; 
		// position i containts num nodes in layer i (first layer is layer 0)

		std::vector<std::vector<double> > weights; 
		// vector containing weight matrixes in row major form (vect of vects)

		const std::vector<Activation> funcs; 
		// vector containing activation func to apply to (W_i)x_i at position i

	public:
		NeuralNet(std::vector<int> l, std::vector<Activation> f); 
		// contructor which initializes data members
		// takes vectors of layer sizes and activation functions as input

		void printWeights(int l) const;		
		// prints the weights
		// l specifies which layers' weights (0 to num_layers - 1)

		void activation(std::vector<double> x, Activation f);
		// applies activation function to x

};

#endif	// NEURALNET_H
