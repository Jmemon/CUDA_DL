#include "../include/Loss.cuh"
#include <vector>
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <stdlib.h>

/* ----------------------------------------------
mse_functor

Defines operator that returns squared difference between x and y
For use with thrust::transform()
---------------------------------------------- */
struct mse_functor
{

	mse_functor() {}

	__host__ __device__ double operator() (const double &x, const double &y) const
	{
		return (x - y) * (x - y);
	} // end operator

}; // end mse_functor

/* ----------------------------------------------
mseGPU

Parameters:
	x - vector of predictions, can be matrix in row-major form 
	y - vector of actual outputs, can be matrix in row-major form
	batch_size - number of samples in batch

Uses thrust device_vector and transform to get error
Uses operation defined in mse_functor with thrust::transform
Outputs (1/(2*batch_size))(sum{norm{y-aL}^2})

Returns:
	mse - averaged mean-square error over all samples in batch
---------------------------------------------- */
double mseGPU(std::vector<double> &x, std::vector<double> &y, int batch_size)
{
	double mse = 0.0;
	thrust::device_vector<double> d_x(x);
	thrust::device_vector<double> d_y(y);

	// y[i] <- (x[i] - y[i]) * (x[i] - y[i])
	thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_y.begin(), mse_functor());
	
	// set mse to be the sum of squares of each element of every sample of d_y
	mse = thrust::reduce(d_y.begin(), d_y.end(), (double) 0.0, thrust::plus<double>());

	// take batch mse average
	mse /= (2 * batch_size);

	return mse;
} // end mseGPU

/* ----------------------------------------------
msePrime

Parameters:
	x - vector representing predicted output, can be matrix in row-major form
	y - vector representing actual output, can be matrix in row-major form
	len - length of x and y

Stores the derivative of mse in y
Derivative for element i: x_i - y_i
---------------------------------------------- */
__global__ void msePrime(double *x, double *y, int len)
{
	int g_idx = gridDim.x * blockDim.x * (blockDim.y * blockIdx.y + threadIdx.y)
		+ blockDim.x * blockIdx.x + threadIdx.x;

	if (g_idx >= len)
		return;

	y[g_idx] = x[g_idx] - y[g_idx];
} // end msePrime

/* ----------------------------------------------
msePrimeGPU

Parameters:
	a - vector of predicted outputs, can be matrix in row-major form
	y - vector of actual outputs, can be matrix in row-major form
	size - number of rows in a/y
	batch_size - number of cols in a/y (number of samples in batch)

Calls msePrime cuda kernel on a and y

Returns:
	dC - vector of derivative of MSE wrt each aL_i
---------------------------------------------- */
std::vector<double> msePrimeGPU(std::vector<double> &a, std::vector<double> &y, int size, int batch_size)
{
	double *d_a, *d_y;
	std::vector<double> dC(a.size());
	int BLOCKSIZE = a.size() >= 512 ? 512 : a.size();

	cudaMalloc((void **) &d_a, a.size() * sizeof(double));
	cudaMalloc((void **) &d_y, y.size() * sizeof(double));

	cudaMemcpy(d_a, a.data(), a.size() * sizeof(double), cudaMemcpyHostToDevice);

	dim3 GRID((a.size() + BLOCKSIZE - 1) / BLOCKSIZE);
	dim3 BLOCK(BLOCKSIZE);

	msePrime<<<GRID, BLOCK, 0>>>(d_a, d_y, size * batch_size);
	cudaDeviceSynchronize();

	cudaMemcpy(dC.data(), d_y, a.size() * sizeof(double), cudaMemcpyDeviceToHost);

	for (int i = 0; i < dC.size(); i++)
		dC[i] /= batch_size;

	cudaFree(d_a);
	cudaFree(d_y);

	return dC;
} // end msePrimeGPU

/* ----------------------------------------------
logLoss_functor

Defines operator that returns -y*log_e(x)
For use with thrust::transform()
---------------------------------------------- */
struct logLoss_functor
{
	
	logLoss_functor() {}

	__host__ __device__ double operator() (const double &x, const double &y) const 
	{
		return -1 * y * log(x);
	}

}; // end ln_functor

/* ----------------------------------------------
crossEntropyGPU

Parameters:
	x - vector of predictions, can be matrix in row-major form 
	y - vector of actual outputs, can be matrix in row-major form
	batch_size - number of samples in batch

Uses thrust device_vector and transform to get error
Uses operation defined in logLoss_functor with thrust::transform
Outputs (1/batch_size)(sum{-y*log_e(x)})

Returns:
	logLoss - cross entropy loss over all samples averaged with batch_size
---------------------------------------------- */
double crossEntropyGPU(std::vector<double> &x, std::vector<double> &y, int batch_size)
{
	double logLoss = 0.0;
	thrust::device_vector<double> d_x(x);
	thrust::device_vector<double> d_y(y);

	// y[i] <- y[i] * log_e(x[i])
	thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_y.begin(), logLoss_functor());

	// logLoss <- sum{y[i] * log_e(x[i])}
	logLoss = thrust::reduce(d_y.begin(), d_y.end(), (double) 0.0, thrust::plus<double>());

	// logLoss <- (1/batch_size) * logLoss
	logLoss /= batch_size;

	return logLoss;
} // end crossEntropyGPU

/* ----------------------------------------------
crossEntropyPrime

Parameters:
	a - vector representing predicted output, can be matrix in row-major form
	y - vector representing actual output, can be matrix in row-major form
	len - length of x and y

Stores the derivative of crossEntropy in y
Derivative for element i: -y_i / a_i
---------------------------------------------- */
__global__ void crossEntropyPrime(double *a, double *y, int len)
{
	int g_idx = gridDim.x * blockDim.x * (blockIdx.y * blockDim.y + threadIdx.y)
		+ blockIdx.x * blockDim.x + threadIdx.x;
	
	if (g_idx >= len)
		return;

	y[g_idx] = -1 * y[g_idx] / a[g_idx];
} // end crossEntropyPrime

/* ----------------------------------------------
crossEntropyPrimeGPU

Parameters:
	a - vector of predicted outputs, can be matrix in row-major form
	y - vector of actual outputs, can be matrix in row-major form
	size - number of rows in a/y
	batch_size - number of cols in a/y (number of samples in batch)

Calls crossEntropyPrime cuda kernel on a and y

Returns:
	dC - vector of derivative of Cross Entropy wrt each aL_i
---------------------------------------------- */
std::vector<double> crossEntropyPrimeGPU(std::vector<double> &a, std::vector<double> &y, int size, int batch_size)
{
	double *d_a, *d_y;
	std::vector<double> dC(a.size());
	int BLOCKSIZE = a.size() >= 512 ? 512 : a.size();

	cudaMalloc((void **) &d_a, a.size() * sizeof(double));
	cudaMalloc((void **) &d_y, y.size() * sizeof(double));

	cudaMemcpy(d_a, a.data(), a.size() * sizeof(double), cudaMemcpyHostToDevice);

	dim3 GRID((a.size() + BLOCKSIZE - 1) / BLOCKSIZE);
	dim3 BLOCK(BLOCKSIZE);

	crossEntropyPrime<<<GRID, BLOCK, 0>>>(d_a, d_y, size * batch_size);
	cudaDeviceSynchronize();

	cudaMemcpy(dC.data(), d_y, a.size() * sizeof(double), cudaMemcpyDeviceToHost);

	for (int i = 0; i < dC.size(); i++)
		dC[i] /= batch_size;

	cudaFree(d_a);
	cudaFree(d_y);

	return dC;
} // end msePrimeGPU
