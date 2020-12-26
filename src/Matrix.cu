#include "../include/Matrix.cuh"

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
   
	if( col < k && row < m) 
	{
		for(int i = 0; i < n; i++) 
			sum += a[row * n + i] * b[i * k + col];
        
		c[row * k + col] = sum;
	} // end if

} // end matMul

void matMulGPU(double *a, double *b, double *c, int m, int n, int k, dim3 Dg, dim3 Dn, size_t Ns)
{
	double *d_a, *d_b, *d_c;
	
	cudaMalloc((void **) &d_a, m * n * sizeof(double));
	cudaMalloc((void **) &d_b, n * k * sizeof(double));
	cudaMalloc((void **) &d_c, m * k * sizeof(double));

	cudaMemcpy(d_a, a, m * n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, n * k * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, m * k * sizeof(double), cudaMemcpyHostToDevice);
	
	matMul<<<Dg, Dn, Ns>>>(d_a, d_b, d_c, m, n, k);
	cudaDeviceSynchronize();

	cudaMemcpy(a, d_a, m * n * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(b, d_b, n * k * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(c, d_c, m * k * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	
} // end matMulGPU

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

void matTransGPU(double *mat_in, double *mat_out, unsigned int rows, unsigned int cols, dim3 Dg, dim3 Dn, size_t Ns)
{
	double *d_in, *d_out;

	size_t SIZE = rows * cols * sizeof(double);

	cudaMalloc((void **) &d_in, SIZE); 
	cudaMalloc((void **) &d_out, SIZE);

	cudaMemcpy(d_in, mat_in, SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(d_out, mat_out, SIZE, cudaMemcpyHostToDevice);

	matTrans<<<Dg, Dn, Ns>>>(mat_in, mat_out, rows, cols);
	cudaDeviceSynchronize();

	cudaMemcpy(mat_in, d_in, SIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy(mat_out, d_out, SIZE, cudaMemcpyDeviceToHost);

	cudaFree(d_in);
	cudaFree(d_out);

} // end matTransGPU
