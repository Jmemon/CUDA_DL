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

	x[idx] = (int)(x[idx] > 0) % 2;		// if val is > 0, its 1 ; if ≤ 0, its 0
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

	x[idx] = 1 / (1 + exp(x[idx]));
} // end sigmoid

std::vector<double> sigmoidGPU(std::vector<double>& z)
{
	double *d_z;
	std::vector<double> a(z.size());
	int BLOCKSIZE = z.size() >= 512 ? 512 : z.size();

	cudaMalloc((void **) &d_z, z.size() * sizeof(double));
	
	cudaMemcpy(d_z, z.data(), z.size() * sizeof(double), cudaMemcpyHostToDevice);

	dim3 GRID((z.size() + BLOCKSIZE - 1) / BLOCKSIZE);
	dim3 BLOCK(BLOCKSIZE);

	sigmoid<<<GRID, BLOCK, 0>>>(d_z, z.size());
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

std::vector<double> reluGPU(std::vector<double>& z)
{
	double *d_z;
	std::vector<double> a(z.size());
	int BLOCKSIZE = z.size() >= 512 ? 512 : z.size();

	cudaMalloc((void **) &d_z, z.size() * sizeof(double));
	
	cudaMemcpy(d_z, z.data(), z.size() * sizeof(double), cudaMemcpyHostToDevice);

	dim3 GRID((z.size() + BLOCKSIZE - 1) / BLOCKSIZE);
	dim3 BLOCK(BLOCKSIZE);

	relu<<<GRID, BLOCK, 0>>>(d_z, z.size());
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

std::vector<double> leakyReluGPU(std::vector<double>& z)
{
	double *d_z;
	std::vector<double> a(z.size());
	int BLOCKSIZE = z.size() >= 512 ? 512 : z.size();

	cudaMalloc((void **) &d_z, z.size() * sizeof(double));
	
	cudaMemcpy(d_z, z.data(), z.size() * sizeof(double), cudaMemcpyHostToDevice);

	dim3 GRID((z.size() + BLOCKSIZE - 1) / BLOCKSIZE);
	dim3 BLOCK(BLOCKSIZE);

	leakyRelu<<<GRID, BLOCK, 0>>>(d_z, z.size());
	cudaDeviceSynchronize();

	cudaMemcpy(a.data(), d_z, z.size() * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_z);

	return a;
} // end leakyReluGPU
