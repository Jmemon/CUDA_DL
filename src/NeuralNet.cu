#include "../include/NeuralNet.h"
#include "../include/Activation.cuh"
#include <vector>
#include <iostream>

NeuralNet::NeuralNet(const vector<int> l, const vector<Activation> f) 
	: num_layers(l.size()), layers(l), funcs(f)
{
	
	for (int i = 0; i < num_layers - 1; i++) 
	{
		vector<double> tmp(l[i] * l[i + 1] , 0);
		
		for (int j = 0; j < tmp.size(); j++)
			tmp[j] = (double)(rand() % 10000) / 10000;

		weights.push_back(tmp);

	} // end for

} // end NeuralNet

/* void NeuralNet::initWeights() {

	int i;
	double** d_w = (double **)malloc((num_layers - 1) * sizeof(double *));

	for (i = 0; i < num_layers - 1; i++) {

		int num_weights = layers[i] * layers[i + 1];
		size_t SIZE = num_weights * sizeof(double);
		
		cudaMalloc((void **)&(*(d_w + i)), SIZE);

		cudaMemcpy(*(d_w + i), *(weights + i), SIZE, cudaMemcpyHostToDevice);

		dim3 BLOCKS(num_weights / 1024 + 1, 1, 1);
		dim3 THREADS(num_weights / (num_weights / 1024 + 1), 1, 1);

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start);
		randInitGPU(*(d_w + i), BLOCKS, THREADS);
		cudaEventRecord(stop);

		cudaEventSynchronize(stop);

		cudaMemcpy(*(weights + i), *(d_w + i), SIZE, cudaMemcpyDeviceToHost);

		float ms;
		cudaEventElapsedTime(&ms, start, stop);

		printf("W%d Init Time: %f \n", i + 1, ms);
		printf("W%d Size: %d x %d \n\n", i + 1, layers[i + 1], layers[i]);

		cudaFree(*(d_w + i));
	}

	free(d_w);

} */

void NeuralNet::printWeights(vector<double> w, int l1, int l2) const 
{	
	for (int i = 0; i < l2; i++) 
	{	
		for (int j = 0; j < l1; j++) 
			std::cout << w[i * l1 + j] << "  ";			
		
		std::cout << std::endl;

	} // end for

	std::cout << std::endl;

} // end printWeights

void NeuralNet::activation(vector<double> x, Activation f) 
{	
	size_t SIZE = x.size() * sizeof(double);

	double *d_x;
	cudaMalloc((void **) &d_x, SIZE);

	cudaMemcpy(d_x, x.data(), SIZE, cudaMemcpyHostToDevice);

	dim3 BLOCKS(len / 1024 + 1, 1, 1);
	dim3 THREADS(len / (len / 1024 + 1), 1, 1);
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	if (f == binary_step) {
		cudaEventRecord(start);
		binaryStepGPU(d_x, BLOCKS, THREADS);	
		cudaEventRecord(stop);
	} /*else if (f == sigmoid) {
		cudaEventRecord(start);
		sigmoid<<<BLOCKS, THREADS>>>(d_x);
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
