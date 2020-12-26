#ifndef LOSS_CUH
#define LOSS_CUH

typedef enum lossType 
{
	mse,
	cross_entropy
} lossType;

// Note that these functions assume at most a 2d grid of blocks

/* -----------------------------------------------
err is where the errors will be stored
e_size is the number of errors to calculate (if x is mxn, e_size = n)
x is where the networks output should go
y is where the actual output should go
size is the length of y (if x is mxn, size = m)
----------------------------------------------- */
void mseGPU(double *err, int e_size, double *x, double *y, int size, dim3 Dg, dim3 Dn, size_t Ns = 0);

void crossEntropyGPU(double *err, int e_size, double *x, double *y, int size, dim3 Dg, dim3 Dn, size_t Ns = 0);

#endif // LOSS_CUH
