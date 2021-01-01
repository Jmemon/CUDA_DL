#include "../include/Matrix.cuh"
#include <vector>
#include <algorithm> // std::max

/* ---------------------------------------------------------------
matMul

Parameters: 
	a - double ptr representing matrix A in row-major form
    b - double ptr representing matrix B in row-major form
    c - double ptr where AB will be stored in row-major form
	m - rows in A / C
	n - cols in A / rows in B
	k - cols in B / C

Multiplies the matrices stored in row-major form in a and b, then stores
	the output in c

Could be optimized much further with shared memory
--------------------------------------------------------------- */
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

/* ---------------------------------------------------------------
matMulGPU

Parameters: 
	a - vector representing first matrix
    b - vector representing second matrix
	m - rows in a
	n - cols in a / rows in b
	k - cols in b

Calls cuda kernel matMul on a.data() and b.data()

Returns:
	c - vector representing AB (has dim m x k)
--------------------------------------------------------------- */
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

/* ---------------------------------------------------------------
scalarMult

Parameters: 
	a - double ptr representing matrix A in row-major form
	c - scalar to multiply a by
	len - int representing length of row-major representation of A

Performs scalar multiplication and stores result in a
--------------------------------------------------------------- */
__global__ void scalarMult(double *a, double c, int len)
{
	int g_idx = gridDim.x * blockDim.x * (blockDim.y * blockIdx.y + threadIdx.y)
		+ blockDim.x * blockIdx.x + threadIdx.x;

	if (g_idx >= len)
		return;

	a[g_idx] = c * a[g_idx];
} // end scalarMult

/* ---------------------------------------------------------------
scalarMultGPU

Parameters: 
	a - vector representing matrix A
	c - scalar to multiply a by
	m - rows in A 
	n - cols in A 

Calls cuda kernel scalarMult on a.data()

Returns:
	B - vector representing cA (has dim m x n)
--------------------------------------------------------------- */
std::vector<double> scalarMultGPU(std::vector<double>& a, double c, int m, int n)
{
	double *d_a;
	std::vector<double> b(m * n);	
	int BLOCKSIZE = m >= 32 || n >= 32 ? 32 : std::max(m, n);

	cudaMalloc((void **) &d_a, m * n * sizeof(double));

	cudaMemcpy(d_a, a.data(), m * n * sizeof(double), cudaMemcpyHostToDevice);
	
	dim3 GRID((n + BLOCKSIZE - 1) / BLOCKSIZE, (m + BLOCKSIZE - 1) / BLOCKSIZE);
	dim3 BLOCK(BLOCKSIZE, BLOCKSIZE);

	scalarMult<<<GRID, BLOCK, 0>>>(d_a, c, m * n);
	cudaDeviceSynchronize();

	cudaMemcpy(b.data(), d_a, m * n * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_a);

	return b;
} // end scalarMultGPU

/* ---------------------------------------------------------------
hadamard

Parameters: 
	a - double ptr representing matrix A in row-major form
    b - double ptr representing matrix B in row-major form
    c - double ptr where A o B will be stored in row-major form
	len - the length of row-major form of A, B, and C

Performs Hadamard operation (element-wise mult) and stores result in c
--------------------------------------------------------------- */
__global__ void hadamard(double *a, double *b, double *c, int len)
{
	int g_idx = gridDim.x * blockDim.x * (blockIdx.y * blockDim.y + threadIdx.y)
		+ blockIdx.x * blockDim.x + threadIdx.x;

	if (g_idx >= len)
		return;

	c[g_idx] = a[g_idx] + b[g_idx];
} // end haramard

/* ---------------------------------------------------------------
hadamardGPU

Parameters: 
	a - vector representing matrix A
    b - vector representing matrix B
	m - rows in A / B
	n - cols in A / B

Calls cuda kernel hadamard on a.data() and b.data()

Returns:
	c - vector representing A o B (has dim m x n)
--------------------------------------------------------------- */
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

/* ---------------------------------------------------------------
matAdd

Parameters: 
	a - double ptr representing matrix A in row-major form
    b - double ptr representing matrix B in row-major form
    c - double ptr where A + B will be stored in row-major form
	len - the length of row-major form of A, B, and C

Performs A + B and stores result in c
--------------------------------------------------------------- */
__global__ void matAdd(double *a, double *b, double *c, int len)
{
	int g_idx = gridDim.x * blockDim.x * (blockIdx.y * blockDim.y + threadIdx.y)
		+ blockIdx.x * blockDim.x + threadIdx.x;
	
	if (g_idx >= len)
		return;

	c[g_idx] = a[g_idx] + b[g_idx];
} // end matAdd

/* ---------------------------------------------------------------
matAddGPU

Parameters: 
	a - vector representing matrix A
    b - vector representing matrix B
	m - rows in A / B
	n - cols in A / B

Calls cuda kernel matAdd on a.data() and b.data()

Returns:
	c - vector representing A + B (has dim m x n)
--------------------------------------------------------------- */
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

/* ---------------------------------------------------------------
matReciprocal

Parameters: 
	a - double ptr representing matrix A in row-major form
	len - length of vector representing A

raises each value in A to the -1 power
--------------------------------------------------------------- */
__global__ void matReciprocal(double *a, int len)
{
	int g_idx = gridDim.x * blockDim.x * (blockDim.y * blockIdx.y + threadIdx.y)
		+ blockDim.x * blockIdx.x + threadIdx.x;
	
	if (len >= g_idx)
		return;

	a[g_idx] = 1.0 / a[g_idx];
} // end matReciprocal

/* ---------------------------------------------------------------
matReciprocalGPU

Parameters: 
	a - vector representing matrix A
	m - rows in matrix A
	n - cols in matrix A

Calls cuda kernel matReciprocal on a.data()

Returns:
	c - vector representing reciprocal A 
--------------------------------------------------------------- */
std::vector<double> matReciprocalGPU(std::vector<double>& a, int m, int n)
{
	double *d_a;
	std::vector<double> c(m * n);
	int BLOCKSIZE = m >= 32 || n >= 32 ? 32 : std::max(m, n);	

	cudaMalloc((void **) &d_a, m * n * sizeof(double));

	cudaMemcpy(d_a, a.data(), m * n * sizeof(double), cudaMemcpyHostToDevice);
	
	dim3 GRID((n + BLOCKSIZE - 1) / BLOCKSIZE, (m + BLOCKSIZE - 1) / BLOCKSIZE);
	dim3 BLOCK(BLOCKSIZE, BLOCKSIZE);

	matReciprocal<<<GRID, BLOCK, 0>>>(d_a, m * n); 
	cudaDeviceSynchronize();

	cudaMemcpy(c.data(), d_a, m * n * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	
	return c;
} // end matMulGPU
/* ---------------------------------------------------------------
matTrans

Parameters: 
	a - double ptr representing matrix A in row-major form
    aT - double ptr representing matrix AT in row-major form
	m - rows in A / cols in AT
	n - cols in A / rows in AT

Transposes matrix A
--------------------------------------------------------------- */
__global__ void matTrans(double *a, double *aT, int m, int n)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col >= n || row >= m)
		return;

	if (col < n && row < m) 
	{
		int pos = row * n + col;
		int trans_pos = col * m + row;
		aT[trans_pos] = a[pos];
	} // end if

} // end matTrans

/* ---------------------------------------------------------------
matTransGPU

Parameters: 
	a - vector representing matrix A
	m - rows in A / cols in AT
	n - cols in A / rows in AT

Calls cuda kernel matTrans on a.data()

Returns:
	aT - vector representing AT
--------------------------------------------------------------- */
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
