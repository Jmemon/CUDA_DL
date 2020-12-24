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

		const vector<int> layers; //int *layers;
		// position i containts num nodes in layer i (first layer is layer 0)

		const vector<vector<double> > weights; //double **weights;
		// vector containing weight matrixes in row major form (vect of vects)

		const vector<Activation> funcs; //Activation *funcs;
		// vector containing activation func to apply to (W_i)x_i at position i

	public:
		NeuralNet(vector<int> l, vector<Activation> f); // int n);	
		// contructor which initializes data members
		// takes vectors of layer sizes and activation functions as input

		void printWeights(vector<double> w, int l1, int l2) const;		
		// prints the weights

		void activation(vector<double> x, int len, Activation f);
		// applies activation function to x

};

#endif	// NEURALNET_H
