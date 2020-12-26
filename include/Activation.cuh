#ifndef ACTIVATION_CUH
#define ACTIVATION_CUH

void binaryStepGPU(double *x, dim3 Dg, dim3 Dn, size_t Ns = 0);

void sigmoidGPU(double *x, dim3 Dg, dim3 Dn, size_t Ns = 0);

void reluGPU(double *x, dim3 Dg, dim3 Dn, size_t Ns = 0);

void leakyReluGPU(double *x, dim3 Dg, dim3 Dn, size_t Ns = 0);

#endif // ACTIVATION_CUH
