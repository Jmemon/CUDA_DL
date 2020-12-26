#include "../include/Loss.cuh"

__global__ void mse(double *err, double *x, double *y)
{
	int idx = gridDim.x * blockDim.x * (blockDim.y * blockIdx.y + threadIdx.y)
		+ blockDim.x * blockIdx.x + threadIdx.x;
	// first line finds number of threads in rows of grid above thread's row 
	// second line finds how far into row thread is

	double sqrErr = (x[idx] - y[idx]) * (x[idx] - y[idx]);
	
} // end mse

void mseGPU(double *err, int e_size, double *x, double *y, int size, dim3 Dg, dim3 Dn, size_t Ns)
{
	double *d_err, *d_x, *d_y;
	
	cudaMalloc((void **) &d_err, e_size * sizeof(double));
	cudaMalloc((void **) &d_x, size * sizeof(double));
	cudaMalloc((void **) &d_y, size * sizeof(double));

	cudaMemcpy(d_err, err, e_size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, size * sizeof(double), cudaMemcpyHostToDevice);

	mse<<<Dg, Dn, Ns>>>(d_err, d_x, d_y);

	cudaMemcpy(err, d_err, e_size * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(x, d_x, size * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(y, d_y, size * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_err);
	cudaFree(d_x);
	cudaFree(d_y);

} // end mseGPU

__global__ void crossEntropy(double *err, double *x, double *y)
{

} // end crossEntropy

void crossEntropyGPU(double *err, int e_size, double *x, double *y, int size, dim3 Dg, dim3 Dn, size_t Ns)
{
	double *d_err, *d_x, *d_y;
	
	cudaMalloc((void **) &d_err, e_size * sizeof(double));
	cudaMalloc((void **) &d_x, size * sizeof(double));
	cudaMalloc((void **) &d_y, size * sizeof(double));

	cudaMemcpy(d_err, err, e_size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, size * sizeof(double), cudaMemcpyHostToDevice);

	crossEntropy<<<Dg, Dn, Ns>>>(d_err, d_x, d_y);

	cudaMemcpy(err, d_err, e_size * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(x, d_x, size * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(y, d_y, size * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_err);
	cudaFree(d_x);
	cudaFree(d_y);


} // end crossEntropyGPU
