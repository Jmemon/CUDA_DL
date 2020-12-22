#include "../include/Activation.cuh"
#include <curand.h>
#include <curand_kernel.h>

__global__ void randInit(double *w) {

	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	curandState_t state;
	curand_init(idx, 0, 0, &state);	// seed, seq num, offset, curandState_t ptr

	w[idx] = (double)(curand(&state) % 10000) / 10000;
}

void randInitGPU(double *x, dim3 Dg, dim3 Dn, size_t Ns) {
	randInit<<<Dg, Dn, Ns>>>(x);
	cudaDeviceSynchronize();
}

__device__ double maxGPU(double a, double b) {
	bool sel = (a <= b);
	return (double)(sel * b - (1 - sel) * a);
}

__global__ void binaryStep(double *x) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	x[idx] = (int)(x[idx] > 0) % 2;		// if val is > 0, its 1 ; if ≤ 0, its 0
}

void binaryStepGPU(double *x, dim3 Dg, dim3 Dn, size_t Ns) {
	binaryStep<<<Dg, Dn, Ns>>>(x);
	cudaDeviceSynchronize();
}

// Need to work out the fractional exponent stuff for sigmoid
/*
__global__ void sigmoid(double *x) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	x[idx] = 1 / (1 + 2.71828)
}*/

__global__ void relu(double *x) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	x[idx] = maxGPU(x[idx], 0);	// if val > 0, return itself ; if ≤ 0, return 0
}

void reluGPU(double *x, dim3 Dg, dim3 Dn, size_t Ns) {
	relu<<<Dg, Dn, Ns>>>(x);
	cudaDeviceSynchronize();
}

__global__ void leakyRelu(double *x) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	x[idx] = maxGPU(x[idx], 0.05 * x[idx]);
}

void leakyReluGPU(double *x, dim3 Dg, dim3 Dn, size_t Ns) {
	leakyRelu<<<Dg, Dn, Ns>>>(x);
	cudaDeviceSynchronize();
}	
