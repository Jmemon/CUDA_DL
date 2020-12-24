#include "../include/NeuralNet.h"
#include "../include/Activation.cuh"
#include <vector>
#include <iostream>

NeuralNet::NeuralNet(const std::vector<int> l, const std::vector<Activation> f) 
	: num_layers(l.size()), layers(l), funcs(f)
{
	
	for (int i = 0; i < num_layers - 1; i++) 
	{
		std::vector<double> tmp(l[i] * l[i + 1] , 0);
		
		for (int j = 0; j < tmp.size(); j++)
			tmp[j] = (double)(rand() % 10000) / 10000;

		weights.push_back(tmp);

	} // end for

} // end NeuralNet

void NeuralNet::printWeights(int l) const 
{

	if (l >= num_layers - 1)
	{
		std::cout << "l is too large" << std::endl;
		return;
	}

	std::cout << "Weights for layers " << l << " to " << l + 1 << ":" << std::endl;
	
	for (int i = 0; i < layers[l + 1]; i++) 
	{	
		for (int j = 0; j < layers[l]; j++) 
			std::cout << weights[l][i * layers[l] + j] << "  ";			
		
		std::cout << std::endl;

	} // end for

	std::cout << std::endl;

} // end printWeights

void NeuralNet::activation(std::vector<double> x, Activation f) 
{	
	size_t SIZE = x.size() * sizeof(double);

	double *d_x;
	cudaMalloc((void **) &d_x, SIZE);

	cudaMemcpy(d_x, x.data(), SIZE, cudaMemcpyHostToDevice);

	dim3 BLOCKS(x.size() / 1024 + 1, 1, 1);
	dim3 THREADS(x.size() / (x.size() / 1024 + 1), 1, 1);
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	if (f == binary_step) {
		cudaEventRecord(start);
		binaryStepGPU(d_x, BLOCKS, THREADS);	
		cudaEventRecord(stop);
	} /*else if (f == sigmoid) {
		cudaEventRecord(start);
		sigmoidGPU(d_x, BLOCKS, THREADS);
		cudaEventRecord(stop);
	}*/ else if (f == relu) {
		cudaEventRecord(start);
		reluGPU(d_x, BLOCKS, THREADS);
		cudaEventRecord(stop);
	} else if (f == leaky_relu) {
		cudaEventRecord(start);
		leakyReluGPU(d_x, BLOCKS, THREADS);
		cudaEventRecord(stop);
	} else {
		std::cout << "Activation function must be binary_step, relu, or leaky_relu" << std::endl;
	}

	cudaEventSynchronize(stop);

	cudaMemcpy(x.data(), d_x, SIZE, cudaMemcpyDeviceToHost);

	float ms;
	cudaEventElapsedTime(&ms, start, stop);

	std::cout << "Activation Time: " << ms << std::endl;

} // end activation
