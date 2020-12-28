#ifndef LOSS_CUH
#define LOSS_CUH

#include <vector>

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

// should be use for regression stuff
// sums (yhat_i - y_i)^2 where i will go through the output nodes
std::vector<double> mseGPU(std::vector<double> x, std::vector<double> y, int size, int e_size); 

// should be used for classification
std::vector<double> crossEntropyGPU(std::vector<double> x, std::vector<double> y, int size, int e_size);

#endif // LOSS_CUH
