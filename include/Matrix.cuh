#ifndef MATRIX_CUH
#define MATRIX_CUH

#include <vector>

std::vector<double> matMulGPU(std::vector<double>& a, std::vector<double>& b, int m, int n, int k);

std::vector<double> hadamardGPU(std::vector<double>& a, std::vector<double>& b, int m, int n);

std::vector<double> matAddGPU(std::vector<double>& a, std::vector<double>& b, int m, int n);

std::vector<double> matTransGPU(std::vector<double>& a, int m, int n);

#endif // MATRIX_CUH
