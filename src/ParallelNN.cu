#include "../include/ParallelNN.cuh"
#include "../include/NeuralNet.h"
#include <curand.h>
#include <curand_kernel.h>

__global__ void randInit(double *w) {

	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	curandState_t state;
	curand_init(idx, 0, 0, &state);	// seed, seq num, offset, curandState_t ptr

	w[idx] = (double)(curand(&state) % 10000) / 10000;
}

void randInitGPU(double *w ) {
	randInit<<<1,1 >>>(w);
	cudaDeviceSynchronize();
}

__device__ double max(double a, double b) {
	bool sel = (a <= b);
	return (double)(sel * b - (1 - sel) * a);
}

__global__ void binary_step(double *x) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	x[idx] = (int)(x[idx] > 0) % 2;		// if val is > 0, its 1 ; if ≤ 0, its 0
}

// Need to work out the fractional exponent stuff for sigmoid
/*
__global__ void sigmoid(double *x) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	x[idx] = 1 / (1 + 2.71828)
}*/

__global__ void relu(double *x) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	x[idx] = max(x[idx], 0);	// if val > 0, return itself ; if ≤ 0, return 0
}

__global__ void leaky_relu(double *x) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	x[idx] = max(x[idx], 0.05 * x[idx]);
}
