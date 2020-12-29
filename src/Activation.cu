#include "../include/Activation.cuh"
#include <vector>

__device__ double maxGPU(double a, double b) 
{
	bool sel = (a <= b);
	return (double)(sel) * b + (double)(1 - sel) * a;
} // end maxGPU

__global__ void binaryStep(double *x, int len) 
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= len)
		return;

	x[idx] = (int)(x[idx] >= 0) % 2;	// if val is ≥ 0, its 1 ; if < 0, its 0
} // end binaryStep

std::vector<double> binaryStepGPU(std::vector<double>& z)
{
	double *d_z;
	std::vector<double> a(z.size());
	int BLOCKSIZE = z.size() >= 512 ? 512 : z.size();

	cudaMalloc((void **) &d_z, z.size() * sizeof(double));
	
	cudaMemcpy(d_z, z.data(), z.size() * sizeof(double), cudaMemcpyHostToDevice);

	dim3 GRID((z.size() + BLOCKSIZE - 1) / BLOCKSIZE);
	dim3 BLOCK(BLOCKSIZE);

	binaryStep<<<GRID, BLOCK, 0>>>(d_z, z.size());
	cudaDeviceSynchronize();

	cudaMemcpy(a.data(), d_z, z.size() * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_z);

	return a;
} // binaryStepGPU

// exp(x) returns e^x ; its a cuda library function
__global__ void sigmoid(double *x, int len)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= len)
		return;

	x[idx] = 1 / (1 + exp(-1 * x[idx]));
} // end sigmoid

__global__ void sigmoid_prime(double *x, int len)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= len)
		return;

	x[idx] = exp(-1 * x[idx]) / ((1 + exp(-1 * x[idx])) * (1 + exp(-1 * x[idx])));
} // end sigmoid_prime

std::vector<double> sigmoidGPU(std::vector<double>& z, bool diff)
{
	double *d_z;
	std::vector<double> a(z.size());
	int BLOCKSIZE = z.size() >= 512 ? 512 : z.size();

	cudaMalloc((void **) &d_z, z.size() * sizeof(double));
	
	cudaMemcpy(d_z, z.data(), z.size() * sizeof(double), cudaMemcpyHostToDevice);

	dim3 GRID((z.size() + BLOCKSIZE - 1) / BLOCKSIZE);
	dim3 BLOCK(BLOCKSIZE);

	if (!diff)
		sigmoid<<<GRID, BLOCK, 0>>>(d_z, z.size());
	else
		sigmoid_prime<<<GRID, BLOCK, 0>>>(d_z, z.size());	

	cudaDeviceSynchronize();

	cudaMemcpy(a.data(), d_z, z.size() * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_z);

	return a;
} // end sigmoidGPU

__global__ void relu(double *x, int len) 
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= len)
		return;

	x[idx] = maxGPU(x[idx], 0);	// if val > 0, return itself ; if ≤ 0, return 0
} // end relu

__global__ void relu_prime(double *x, int len) 
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= len)
		return;

	x[idx] = (x[idx] > 0);	// if val > 0, 1 ; if ≤ 0, 0
} // end relu

std::vector<double> reluGPU(std::vector<double>& z, bool diff)
{
	double *d_z;
	std::vector<double> a(z.size());
	int BLOCKSIZE = z.size() >= 512 ? 512 : z.size();

	cudaMalloc((void **) &d_z, z.size() * sizeof(double));
	
	cudaMemcpy(d_z, z.data(), z.size() * sizeof(double), cudaMemcpyHostToDevice);

	dim3 GRID((z.size() + BLOCKSIZE - 1) / BLOCKSIZE);
	dim3 BLOCK(BLOCKSIZE);

	if (!diff)
		relu<<<GRID, BLOCK, 0>>>(d_z, z.size());
	else
		relu_prime<<<GRID, BLOCK, 0>>>(d_z, z.size());

	cudaDeviceSynchronize();

	cudaMemcpy(a.data(), d_z, z.size() * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_z);

	return a;
} // end reluGPU

__global__ void leakyRelu(double *x, int len) 
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= len)
		return;

	x[idx] = maxGPU(x[idx], 0.05 * x[idx]);
} // end leakyRelu

__global__ void leakyRelu_prime(double *x, int len) 
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= len)
		return;

	bool size = x[idx] > 0;

	x[idx] = size * x[idx] + (1 - size) * 0.05;
} // end leakyRelu

std::vector<double> leakyReluGPU(std::vector<double>& z, bool diff)
{
	double *d_z;
	std::vector<double> a(z.size());
	int BLOCKSIZE = z.size() >= 512 ? 512 : z.size();

	cudaMalloc((void **) &d_z, z.size() * sizeof(double));
	
	cudaMemcpy(d_z, z.data(), z.size() * sizeof(double), cudaMemcpyHostToDevice);

	dim3 GRID((z.size() + BLOCKSIZE - 1) / BLOCKSIZE);
	dim3 BLOCK(BLOCKSIZE);

	if (!diff)
		leakyRelu<<<GRID, BLOCK, 0>>>(d_z, z.size());
	else
		leakyRelu_prime<<<GRID, BLOCK, 0>>>(d_z, z.size());

	cudaDeviceSynchronize();

	cudaMemcpy(a.data(), d_z, z.size() * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_z);

	return a;
} // end leakyReluGPU
