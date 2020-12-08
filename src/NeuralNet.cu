#include "../include/NeuralNet.h"
#include <stdlib.h>
#include <stdio.h>

NeuralNet::NeuralNet(int *l, Activation *f, int n) {
	
	num_layers = n;
	layers = l;
	funcs = f;

	weights = (double **)malloc((num_layers - 1) * sizeof(double *));
	for (int i = 0; i < num_layers - 1; i++) {
		*(weights + i) = (double *)malloc(layers[i] * layers[i + 1] * sizeof(double));
	}

	initWeights();

}

void NeuralNet::initWeights() {

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
		randInit<<<BLOCKS, THREADS>>>(*(d_w + i));
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

}

void NeuralNet::printWeights(double* w, int l1, int l2) const {
	
	for (int i = 0; i < l2; i++) {
		
		for (int j = 0; j < l1; j++) {
			printf("%f  ", w[i * l1 + j]);
		}
		printf("\n");

	}
	printf("\n");

}

void NeuralNet::activation(double *x, int len, Activation f) {
	
	size_t SIZE = len * sizeof(double)

	double *d_x;
	cudaMalloc((void **) &d_x, SIZE);

	cudaMemcpy(d_x, x, SIZE, cudaMemcpyHostToDevice);

	dim3 BLOCKS(len / 1024 + 1, 1, 1);
	dim3 THREADS(len / (len / 1024 + 1), 1, 1);
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	if (f == binary_step) {
		cudaEventRecord(start);
		binary_step<<<BLOCKS, THREADS>>>(d_x);
		cudaEventRecord(stop);
	} /*else if (f == sigmoid) {
		cudaEventRecord(start);
		sigmoid<<<BLOCKS, THREADS>>>(d_x);
		cudaEventRecord(stop);
	*/} else if (f == relu) {
		cudaEventRecord(start);
		relu<<<BLOCKS, THREADS>>>(d_x);
		cudaEventRecord(stop);
	} else if (f == leaky_relu) {
		cudaEventRecord(start);
		leaky_relu<<<BLOCKS, THREADS>>>(d_x);
		cudaEventRecord(stop);
	} else {
		cout << "Activation function must be binary_step, relu, or leaky_relu" << endl;"
	}

	cudaEventSynchronize(stop);

	cudaMemcpy(x, d_x, SIZE, cudaMemcpyDeviceToHost);

	double ms;
	cudaEventElapsedTime(&ms, start, stop);

	cout << "Activation Time: " << ms << endl;
}
