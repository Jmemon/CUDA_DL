#ifndef MATRIX_CUH
#define MATRIX_CUH

#include <vector>

// More detailed information on each function is in src/Matrix.cu

// calls matrix multiplication cuda kernel
std::vector<double> matMulGPU(std::vector<double>& a, std::vector<double>& b, int m, int n, int k);

// calls hadamard cuda kernel
std::vector<double> hadamardGPU(std::vector<double>& a, std::vector<double>& b, int m, int n);

// calls matrix addition cuda kernel
std::vector<double> matAddGPU(std::vector<double>& a, std::vector<double>& b, int m, int n);

// calls matrix tranpose cuda kernel
std::vector<double> matTransGPU(std::vector<double>& a, int m, int n);

#endif // MATRIX_CUH
