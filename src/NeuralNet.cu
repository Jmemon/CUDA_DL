#include "../include/NeuralNet.h"
#include "../include/Activation.cuh"
#include "../include/Matrix.cuh"
#include "../include/Loss.cuh"
#include <iostream>
#include <stdio.h>
#include <exception>
#include <vector>

/* -------------------------------------------------- 
Constructor 

Parameters: 
	l - l[i] is number of neurons in layer i
	f - f[i] is activation func for layer i + 1
	e - loss function

Initializes layers to l, funcs to f, errFunc to e
Randomly initializes weights to values between 0 and 1, 
	We determine the sizes of the weight matrices using the values given in l
-------------------------------------------------- */
NeuralNet::NeuralNet(std::vector<int> &l, std::vector<Activation> &f, Loss e) 
	: layers(l), funcs(f), errFunc(e)
{
	if (l.size() < 2)
		throw std::length_error("Network must at least have input and output layer");
	
	if (f.size() != l.size() - 1)
		throw std::length_error("Every layer must have activation except input");

	for (int i = 0; i < layers.size() - 1; i++) 
	{
		std::vector<double> tmp(l[i] * l[i + 1] , 0);
		
		for (int j = 0; j < tmp.size(); j++)
			tmp[j] = (double)(rand() % 10000) / 10000;

		weights.push_back(tmp);

	} // end for

} // end NeuralNet

/* -------------------------------------------------- 
activation

Parameters: 
	x - vector to apply activation func to 
	f - activation func to apply to x

Uses the cuda kernels defined in Activation.cu on x

Returns:
	a - vector equal to f(x)
-------------------------------------------------- */
std::vector<double> NeuralNet::activation(std::vector<double> &x, Activation f) 
{
	if (x.size() < 1)
		throw std::length_error("Layer must have at least one node");

	std::vector<double> a(x.size());

	switch(f)
	{
		case binary_step:
			a = binaryStepGPU(x);
			break;
		case sigmoid:
			a = sigmoidGPU(x);
			break;
		case relu:
			a = reluGPU(x);
			break;
		case leaky_relu:
			a = leakyReluGPU(x);
			break;
		default:
			throw std::domain_error("This activation functions is not implemented.");
	} // end switch

	return a;
} // end activation

/* -------------------------------------------------- 
forwardPass

Parameters: 
	x - vector to apply activation func to
	  - can be a matrix in Row-Major form 

Puts input x through the network and generates a prediction
If x is a matrix where each column is one input, it will do all as a batch
Stores each layer's unactivated value and activated one too for output layer

Returns:
	out - vector with the intermediate values and output 
		- if network is n-m-k and batch_size is 5, then out will have
		5m + 5k + 5k elements, 5 because of the batchsize
-------------------------------------------------- */
std::vector<double> NeuralNet::forwardPass(std::vector<double> &x)
{
	// -- Error Check --------------------------------------------------------
	double tmp = (double)(x.size()) / (double)(layers[0]);

	if (tmp < 1.0)
	{
		char msg [100];
		std::sprintf(msg, "User Input Size: %lud ; NN Input Size: %d", x.size(), layers[0]);
		throw std::length_error(msg);
	} // end if
	
	if (std::floor(tmp) != tmp)
	{
		char msg [100];
		std::sprintf(msg, "Too many/few Input Args (in_size / nn_in_size = %f)", tmp);
		throw std::length_error(msg);
	} // end if
	// ----------------------------------------------------------------------

	int batch_size = x.size() / layers[0];  // num cols in x
	int input_size = layers[0];				// num rows in x
	std::vector<double> out; 	
	std::vector<double> tmpv(x);

	for (int i = 0; i < layers.size() - 1; i++)
	{
		tmpv = matMulGPU(weights[i], tmpv, layers[i + 1], layers[i], batch_size);
		// tmpv = z_(i + 1)
		// weights[i] is layers[i + 1] x layers[i]
		// x is layers[i] x batch_size
		// tmp is layers[i + 1] x batch_size

		for (int j = 0; j < tmpv.size(); j++)
			out.push_back(tmpv[j]);

		tmpv = activation(tmpv, funcs[i]);	
		// tmpv = a_(i + 1)

	} // end for

	for (int i = 0; i < tmpv.size(); i++)
		out.push_back(tmpv[i]);

	return out;
} // end forwardPass

/* -------------------------------------------------- 
calcLoss

Parameters: 
	x - vector of predicted outputs 
	  - can be a matrix in Row-Major form 
	y - vector of actual outputs
	  - can be a matrix in Row-Major form

Applies whatever loss function is specified by NeuralNet.errFunc

Returns:
	err - double which is average error for batch of inputs
-------------------------------------------------- */
double NeuralNet::calcLoss(std::vector<double>& x, std::vector<double>& y)
{
	// -- Error Check --------------------------------------------------------
	double tmp1 = (double)(x.size()) / (double)(layers[layers.size() - 1]);
	double tmp2 = (double)(y.size()) / (double)(layers[layers.size() - 1]);

	if (tmp1 < 1.0) 
	{
		char msg [100];
		std::sprintf(msg, "User Output Size: %lud ; NN Output Size: %d", x.size(), layers[layers.size() - 1]);
		throw std::length_error(msg);
	} // end if
	
	if (tmp2 < 1.0) 
	{
		char msg [100];
		std::sprintf(msg, "User Output Size: %lud ; NN Output Size: %d", y.size(), layers[layers.size() - 1]);
		throw std::length_error(msg);
	} // end if

	if (std::floor(tmp1) != tmp1)
	{
		char msg [100];
		std::sprintf(msg, "Too many/few Input Args (out_size / nn_out_size = %f)", tmp1);
		throw std::length_error(msg);
	} // end if

	if (std::floor(tmp2) != tmp2)
	{
		char msg [100];
		std::sprintf(msg, "Too many/few Input Args (out_size / nn_out_size = %f)", tmp2);
		throw std::length_error(msg);
	} // end if
	// ----------------------------------------------------------------------

	double err;

	switch (errFunc)
	{
		case mse:
			err = mseGPU(x, y, x.size() / layers[layers.size() - 1]);
			break;
		case logLoss:
			err = crossEntropyGPU(x, y, x.size() / layers[layers.size() - 1]);
			break;
		default:
			throw std::domain_error("This loss function has not been implemented");
	} // end switch

	return err;
} // end error

/* -------------------------------------------------- 
printNN

Prints the size of each layer, the activation function
	at each layer, and the loss function at the end of the network
-------------------------------------------------- */
void NeuralNet::printNN() const
{
	std::cout << "Layer 0: " << layers[0] << std::endl;	

	for (int i = 1; i < layers.size(); i++)
	{
		std::cout << std::endl;
		std::cout << "Layer " << i << ": " << layers[i] << std::endl;
		
		switch (funcs[i - 1])
		{
			case binary_step: 
				std::cout << "Activation: Binary Step" << std::endl;
				break;
			case sigmoid: 
				std::cout << "Activation: Sigmoid" << std::endl;
				break;
			case relu: 
				std::cout << "Activation: ReLU" << std::endl;
				break;
			case leaky_relu: 
				std::cout << "Activation: Leaky ReLU" << std::endl;
				break;
			default:
				throw std::domain_error("This activation function is not implemented");
		} // end switch

	} // end for

	std::cout << std::endl;

	switch(errFunc)
	{
		case mse:
			std::cout << "Loss Function: Mean-Squared Error" << std::endl;
			break;
		case logLoss:
			std::cout << "Loss Function: Cross Entropy" << std::endl;
			break;
		default:
			throw std::domain_error("This loss function is not implemented");
	} // end switch

	std::cout << std::endl;

} // end printNN

/* -------------------------------------------------- 
printWeights

Parameter:
	l - layers to print weights for (range is 0 to layers.size() - 2)

Prints all the weights from layer l to l + 1
-------------------------------------------------- */
void NeuralNet::printWeights(int l) const 
{
	if (l < 0)
		throw std::length_error("Not a layer");

	if (l > layers.size() - 2)
		throw std::domain_error("There are no weights for this layer");

	std::cout << "Weights for layers " << l << " to " << l + 1 << ":" << std::endl;
	
	for (int i = 0; i < layers[l + 1]; i++) 
	{	
		for (int j = 0; j < layers[l]; j++) 
			std::cout << weights[l][i * layers[l] + j] << "  ";			
		
		std::cout << std::endl;

	} // end for

	std::cout << std::endl;

} // end printWeights

