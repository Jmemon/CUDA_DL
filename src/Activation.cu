#include "../include/Activation.cuh"
#include <curand.h>
#include <curand_kernel.h>

__device__ double maxGPU(double a, double b) {
	bool sel = (a <= b);
	return (double)(sel) * b + (double)(1 - sel) * a;
}

__global__ void binaryStep(double *x) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	x[idx] = (int)(x[idx] > 0) % 2;		// if val is > 0, its 1 ; if ≤ 0, its 0
}

void binaryStepGPU(double *x, dim3 Dg, dim3 Dn, size_t Ns) {
	binaryStep<<<Dg, Dn, Ns>>>(x);
	cudaDeviceSynchronize();
}

// exp(x) returns e^x ; its a cuda library function
__global__ void sigmoid(double *x) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	x[idx] = 1 / (1 + exp(x[idx]));
}

void sigmoidGPU(double *x, dim3 Dg, dim3 Dn, size_t Ns) {
	sigmoid<<<Dg, Dn, Ns>>>(x);
	cudaDeviceSynchronize();
}

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
