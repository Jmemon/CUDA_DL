#include "../../include/kernels/activation.cuh"
#include <vector>

/* ----------------------------------------------
maxGPU

Parameters:
	a - double
	b - double

Finds max of a and b and returns it

Returns:
	max(a, b)
---------------------------------------------- */
__device__ double maxGPU(double a, double b) 
{
	bool sel = (a <= b);
	return (double)(sel) * b + (double)(1 - sel) * a;
} // end maxGPU

/* ----------------------------------------------
sigmoid

Parameters:
	x - vector to apply activation to, can be matrix in row-major form
	len - length of x

Applies sigmoid (1/(1 + exp(-x))) to every element of x
---------------------------------------------- */
// exp(x) returns e^x ; its a cuda library function
__global__ void sigmoid(double *x, int len)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= len)
		return;

	x[idx] = 1 / (1 + exp(-1 * x[idx]));
} // end sigmoid

/* ----------------------------------------------
sigmoid_prime

Parameters:
	x - vector to apply activation to, can be matrix in row-major form
	len - length of x

Applies sigmoidPrime (exp(-x)(1 + exp(-x))^(-2)) to every element of x
---------------------------------------------- */
__global__ void sigmoid_prime(double *x, int len)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= len)
		return;

	x[idx] = exp(-1 * x[idx]) / ((1 + exp(-1 * x[idx])) * (1 + exp(-1 * x[idx])));
} // end sigmoid_prime

/* ----------------------------------------------
sigmoidGPU

Parameters:
	z - vector to apply activation to, can be matrix in row-major form
	diff - bool determining whether to applu sig or sig_prime

calls sigmoid or sigmoid_prime cuda kernel on z.data()

Returns:
	a - activated z, (f(z) or f'(z))
---------------------------------------------- */
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

/* ----------------------------------------------
relu

Parameters:
	x - vector to apply activation to, can be matrix in row-major form
	len - length of x

Applies relu (x if x > 0, else 0) to every element of x
---------------------------------------------- */
__global__ void relu(double *x, int len) 
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= len)
		return;

	x[idx] = maxGPU(x[idx], 0);	// if val > 0, return itself ; if ≤ 0, return 0
} // end relu

/* ----------------------------------------------
relu_prime

Parameters:
	x - vector to apply activation to, can be matrix in row-major form
	len - length of x

Applies relu_prime (1 if x > 0, else 0) to every element of x
---------------------------------------------- */
__global__ void relu_prime(double *x, int len) 
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= len)
		return;

	x[idx] = (x[idx] > 0);	// if val > 0, 1 ; if ≤ 0, 0
} // end relu

/* ----------------------------------------------
reluGPU

Parameters:
	z - vector to apply activation to, can be matrix in row-major form
	diff - bool determining whether to applu sig or sig_prime

calls relu or relu_prime cuda kernel on z.data()

Returns:
	a - activated z, (f(z) or f'(z))
---------------------------------------------- */
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

/* ----------------------------------------------
leakyRelu

Parameters:
	x - vector to apply activation to, can be matrix in row-major form
	len - length of x

Applies leakyRelu (x if x > 0, else 0.05x) to every element of x
---------------------------------------------- */
__global__ void leakyRelu(double *x, int len) 
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= len)
		return;

	x[idx] = maxGPU(x[idx], 0.05 * x[idx]);
} // end leakyRelu

/* ----------------------------------------------
leakyRelu_prime

Parameters:
	x - vector to apply activation to, can be matrix in row-major form
	len - length of x

Applies leakyRelu_prime (1 if x > 0, else 0.05) to every element of x
---------------------------------------------- */
__global__ void leakyRelu_prime(double *x, int len) 
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= len)
		return;

	bool size = x[idx] > 0;

	x[idx] = size * x[idx] + (1 - size) * 0.05;
} // end leakyRelu

/* ----------------------------------------------
leakyReluGPU

Parameters:
	z - vector to apply activation to, can be matrix in row-major form
	diff - bool determining whether to applu sig or sig_prime

calls leakyRelu or leakyRelu_prime cuda kernel on z.data()

Returns:
	a - activated z, (f(z) or f'(z))
---------------------------------------------- */
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

__global__ void exponential(double *x, int len)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= len)
		return;

	x[idx] = exp(x[idx]);
} // end exponential

std::vector<double> exponentialGPU(std::vector<double>& z, bool diff)
{
	double *d_z;
	std::vector<double> a(z.size());
	int BLOCKSIZE = z.size() >= 512 ? 512 : z.size();

	cudaMalloc((void **) &d_z, z.size() * sizeof(double));
	
	cudaMemcpy(d_z, z.data(), z.size() * sizeof(double), cudaMemcpyHostToDevice);

	dim3 GRID((z.size() + BLOCKSIZE - 1) / BLOCKSIZE);
	dim3 BLOCK(BLOCKSIZE);

	if (!diff)
		exponential<<<GRID, BLOCK, 0>>>(d_z, z.size());
	else
		exponential<<<GRID, BLOCK, 0>>>(d_z, z.size());

	cudaDeviceSynchronize();

	cudaMemcpy(a.data(), d_z, z.size() * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_z);

	return a;
} // end exponentialGPU
