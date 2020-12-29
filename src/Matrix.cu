#include "../include/Matrix.cuh"
#include <vector>
#include <algorithm> // std::max

// https://github.com/lzhengchun/matrix-cuda/blob/master/matrix_cuda.cu 
/*
Parameters: 
            &a GPU device pointer to a m X n matrix (A)
            &b GPU device pointer to a n X k matrix (B)
            &c GPU device output purpose pointer to a m X k matrix (C) 
            to store the result
Note:
    grid and block should be configured as:
        dim3 dimGrid((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    further speedup can be obtained by using shared memory to decrease global memory access times
return: none
*/
// Note Grid gets bigger than 1x1 only if k or m is bigger than BLOCK_SIZE
__global__ void matMul(double *a, double *b, double *c, int m, int n, int k)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y; 
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	double sum = 0.0;
   
	if (row >= m || col >= k)
		return;

	if(col < k && row < m) 
	{
		for(int i = 0; i < n; i++) 
			sum += a[row * n + i] * b[i * k + col];
        
		c[row * k + col] = sum;
	} // end if

} // end matMul

std::vector<double> matMulGPU(std::vector<double>& a, std::vector<double>& b, int m, int n, int k)
{
	double *d_a, *d_b, *d_c;
	std::vector<double> c(m * k);
	int BLOCKSIZE = m >= 32 || k >= 32 ? 32 : std::max(m, k);	

	cudaMalloc((void **) &d_a, m * n * sizeof(double));
	cudaMalloc((void **) &d_b, n * k * sizeof(double));
	cudaMalloc((void **) &d_c, m * k * sizeof(double));

	cudaMemcpy(d_a, a.data(), m * n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b.data(), n * k * sizeof(double), cudaMemcpyHostToDevice);
	
	dim3 GRID((k + BLOCKSIZE - 1) / BLOCKSIZE, (m + BLOCKSIZE - 1) / BLOCKSIZE);
	dim3 BLOCK(BLOCKSIZE, BLOCKSIZE);

	matMul<<<GRID, BLOCK, 0>>>(d_a, d_b, d_c, m, n, k);
	cudaDeviceSynchronize();

	cudaMemcpy(c.data(), d_c, m * k * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	
	return c;
} // end matMulGPU

__global__ void hadamard(double *a, double *b, double *c, int len)
{
	int g_idx = gridDim.x * blockDim.x * (blockIdx.y * blockDim.y + threadIdx.y)
		+ blockIdx.x * blockDim.x + threadIdx.x;

	if (g_idx >= len)
		return;

	c[g_idx] = a[g_idx] + b[g_idx];
} // end haramard

std::vector<double> hadamardGPU(std::vector<double>& a, std::vector<double>& b, int m, int n)
{
	double *d_a, *d_b, *d_c;
	std::vector<double> c(m * n);	
	int BLOCKSIZE = m >= 32 || n >= 32 ? 32 : std::max(m, n);

	cudaMalloc((void **) &d_a, m * n * sizeof(double));
	cudaMalloc((void **) &d_b, m * n * sizeof(double));
	cudaMalloc((void **) &d_c, m * n * sizeof(double));

	cudaMemcpy(d_a, a.data(), m * n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b.data(), m * n * sizeof(double), cudaMemcpyHostToDevice);
	
	dim3 GRID((n + BLOCKSIZE - 1) / BLOCKSIZE, (m + BLOCKSIZE - 1) / BLOCKSIZE);
	dim3 BLOCK(BLOCKSIZE, BLOCKSIZE);

	hadamard<<<GRID, BLOCK, 0>>>(d_a, d_b, d_c, m * n);
	cudaDeviceSynchronize();

	cudaMemcpy(c.data(), d_c, m * n * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return c;
} // end hadamardGPU

__global__ void matAdd(double *a, double *b, double *c, int len)
{
	int g_idx = gridDim.x * blockDim.x * (blockIdx.y * blockDim.y + threadIdx.y)
		+ blockIdx.x * blockDim.x + threadIdx.x;
	
	if (g_idx >= len)
		return;

	c[g_idx] = a[g_idx] + b[g_idx];
} // end matAdd

std::vector<double> matAddGPU(std::vector<double>& a, std::vector<double>& b, int m, int n)
{
	double *d_a, *d_b, *d_c;
	std::vector<double> c(m * n);	
	int BLOCKSIZE = m >= 32 || n >= 32 ? 32 : std::max(m, n);

	cudaMalloc((void **) &d_a, m * n * sizeof(double));
	cudaMalloc((void **) &d_b, m * n * sizeof(double));
	cudaMalloc((void **) &d_c, m * n * sizeof(double));

	cudaMemcpy(d_a, a.data(), m * n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b.data(), m * n * sizeof(double), cudaMemcpyHostToDevice);
	
	dim3 GRID((n + BLOCKSIZE - 1) / BLOCKSIZE, (m + BLOCKSIZE - 1) / BLOCKSIZE);
	dim3 BLOCK(BLOCKSIZE, BLOCKSIZE);

	matAdd<<<GRID, BLOCK, 0>>>(d_a, d_b, d_c, m * n);
	cudaDeviceSynchronize();

	cudaMemcpy(c.data(), d_c, m * n * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	
	return c;
} // end matAddGPU

// https://github.com/lzhengchun/matrix-cuda/blob/master/matrix_cuda.cu 
/*
parameters: 
            &mat_in GPU device pointer to a rows X cols matrix
            &mat_out GPU device output purpose pointer to a cols X rows matrix 
            to store the result
Note:
    grid and block should be configured as:
        dim3 dim_grid((n - 1) / BLOCK_SIZE + 1, (n - 1) / BLOCK_SIZE + 1, 1);
        dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);
return: none
*/
__global__ void matTrans(double *mat_in, double *mat_out, unsigned int rows, unsigned int cols)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx < cols && idy < rows) 
	{
		unsigned int pos = idy * cols + idx;
		unsigned int trans_pos = idx * rows + idy;
		mat_out[trans_pos] = mat_in[pos];
	} // end if

} // end matTrans

std::vector<double> matTransGPU(std::vector<double>& a, int m, int n)
{
	double *d_a, *d_aT;
	std::vector<double> aT(m * n);
	int BLOCKSIZE = m >= 32 || n >= 32 ? 32 : std::max(m, n);

	size_t SIZE = m * n * sizeof(double);

	cudaMalloc((void **) &d_a, SIZE); 
	cudaMalloc((void **) &d_aT, SIZE);

	cudaMemcpy(d_a, a.data(), SIZE, cudaMemcpyHostToDevice);

	dim3 GRID((n + BLOCKSIZE - 1) / BLOCKSIZE, (m + BLOCKSIZE - 1) / BLOCKSIZE);
	dim3 BLOCK(BLOCKSIZE, BLOCKSIZE);

	matTrans<<<GRID, BLOCK, 0>>>(d_a, d_aT, m, n);
	cudaDeviceSynchronize();

	cudaMemcpy(aT.data(), d_aT, SIZE, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_aT);

	return aT;
} // end matTransGPU
