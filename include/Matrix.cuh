#ifndef MATRIX_CUH
#define MATRIX_CUH

void matMulGPU(double *a, double *b, double *c, int m, int n, int k, dim3 Dg, dim3 Dn, size_t Ns = 0);

void matTransGPU(double *mat_in, double *mat_out, int rows, int cols, dim3 Dg, dim3 Dn, size_t Ns = 0);

#endif // MATRIX_CUH
