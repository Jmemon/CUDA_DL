#include "../include/NeuralNet.h"
#include "../include/Activation.cuh"
#include "../include/Matrix.cuh"
#include <iostream>
#include <stdio.h>
#include <exception>
#include <vector>

NeuralNet::NeuralNet(std::vector<int> &l, std::vector<Activation> &f) 
	: layers(l), funcs(f)
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

void NeuralNet::activation(std::vector<double> &x, Activation f) 
{
	if (x.size() < 1)
		throw std::length_error("Layer must have at least one node");

	size_t SIZE = x.size() * sizeof(double);

	float ms;
	double *d_x;
	cudaMalloc((void **) &d_x, SIZE);

	cudaMemcpy(d_x, x.data(), SIZE, cudaMemcpyHostToDevice);
	
	dim3 GRID((x.size() - 1) / 1024 + 1);
	dim3 BLOCK(1024);
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	switch(f)
	{
		case binary_step:
			cudaEventRecord(start);
			binaryStepGPU(d_x, GRID, BLOCK);	
			cudaEventRecord(stop);
			break;
		case sigmoid:
			cudaEventRecord(start);
			sigmoidGPU(d_x, GRID, BLOCK);	
			cudaEventRecord(stop);
			break;
		case relu:
			cudaEventRecord(start);
			reluGPU(d_x, GRID, BLOCK);	
			cudaEventRecord(stop);
			break;
		case leaky_relu:
			cudaEventRecord(start);
			leakyReluGPU(d_x, GRID, BLOCK);	
			cudaEventRecord(stop);
			break;
		default:
			throw std::domain_error("This activation functions is not implemented.");
	} // end switch

	// cudaEventSynchronize(stop);
	// waits until everything that began during stop's duration has completed
	// not needed bc each of these gpu funcs use cudaDeviceSynchronize() after kernel call instead

	cudaMemcpy(x.data(), d_x, SIZE, cudaMemcpyDeviceToHost);
	
	cudaEventElapsedTime(&ms, start, stop);
	
	std::cout << "Activation Time: " << ms << std::endl;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_x);

} // end activation

void NeuralNet::forwardPass(std::vector<double> &x)
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
	int BLOCK_SIZE = 32;
	
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	
	for (int i = 0; i < layers.size() - 1; i++)
	{
		dim3 grid((batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE, (layers[i + 1] + BLOCK_SIZE - 1) / BLOCK_SIZE);

		matMulGPU(weights[i].data(), x.data(), x.data(), layers[i + 1], layers[i], batch_size, grid, block);
		// weights[i] is layers[i + 1] x layers[i]
		// x is layers[i] x batch_size
		// tmp is layers[i + 1] x batch_size

		x.resize(layers[i + 1] * batch_size);

		activation(x, funcs[i]);	

	} // end for

} // end forwardPass

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

} // end printNN

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

