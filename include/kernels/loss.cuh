#ifndef LOSS_CUH
#define LOSS_CUH

#include <vector>

// More detailed information about functions in src/Loss.cu

// Note that these functions assume at most a 2d grid of blocks

/* -----------------------------------------------
x is where the networks output should go
y is where the actual output should go
batch_size is number of samples in batch
----------------------------------------------- */

// should be use for regression stuff
// (1/2n)sum[norm(yhat_i - y_i)^2] 
// where i will go through each prediction-actual pair
// n is the batch size
double mseGPU(std::vector<double> &x, std::vector<double> &y, int batch_size); 

// mse derivative
// outputs vector where each elem is deriv of mse wrt corresponding entry of aL
// deriv: dC/daL_i = (1/batch_size) (y_i - aL_i)
std::vector<double> mseDerivativeGPU(std::vector<double> &x, std::vector<double> &y, int size, int batch_size);

// should be used for classification
double crossEntropyGPU(std::vector<double> &x, std::vector<double> &y, int batch_size);

// logLoss derivative
// outputs vector where each elem is deriv of logLoss wrt corresponding entry of aL
// deriv: dC/daL_i = (1/batch_size) (y_i / aL_i)
std::vector<double> crossEntropyDerivativeGPU(std::vector<double> &a, std::vector<double> &y, int size, int batch_size);

#endif // LOSS_CUH
