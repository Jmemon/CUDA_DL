#include "../include/NeuralNet.h"
#include "../include/Activation.cuh"
#include "../include/Matrix.cuh"
#include "../include/Loss.cuh"
#include <iostream>
#include <cmath>
#include <stdio.h>
#include <exception>
#include <vector>

/* -------------------------------------------------- 
Constructor 

Parameters: 
	l - l[i] is number of neurons in layer i
	f - f[i] is activation func for layer i + 1
	e - loss function

Initializes layers to l, funcs to f, errFunc to e
Randomly initializes weights to values between 0 and 1, 
	We determine the sizes of the weight matrices using the values given in l
-------------------------------------------------- */
NeuralNet::NeuralNet(std::vector<int> &l, std::vector<Activation> &f, Loss e, LROptim lr) 
	: layers(l), funcs(f), errFunc(e), alphaAlg(lr), adamIters(0)
{
	if (l.size() < 2)
		throw std::length_error("Network must at least have input and output layer");
	
	if (f.size() != l.size() - 1)
		throw std::length_error("Every layer must have activation except input");

	for (int i = 0; i < layers.size() - 1; i++) 
	{
		std::vector<double> tmpW(l[i] * l[i + 1] , 0.0);
		std::vector<double> tmpMV(l[i] * l[i + 1], 0.0);
	
		for (int j = 0; j < tmpW.size(); j++)
			tmpW[j] = (double)(rand() % 10000) / 10000;

		weights.push_back(tmpW);
		gradWeightedMean.push_back(tmpMV);
		gradBiasedVariance.push_back(tmpMV);
	} // end for

} // end NeuralNet

/* -------------------------------------------------- 
activation

Parameters: 
	x - vector to apply activation func to 
	f - activation func to apply to x

Uses the cuda kernels defined in Activation.cu on x

Returns:
	a - vector equal to f(x)
-------------------------------------------------- */
std::vector<double> NeuralNet::activation(std::vector<double> &x, Activation f, bool diff) 
{
	if (x.size() < 1)
		throw std::length_error("Layer must have at least one node");

	std::vector<double> a(x.size());

	switch(f)
	{
		case binary_step:

			if (diff)
				throw std::domain_error("activation: No binStep deriv implemented");

			a = binaryStepGPU(x);
			break;
		case sigmoid:
			a = sigmoidGPU(x, diff);
			break;
		case relu:
			a = reluGPU(x, diff);
			break;
		case leaky_relu:
			a = leakyReluGPU(x, diff);
			break;
		case exponential:
			a = exponentialGPU(x, diff);
			break;
		default:
			throw std::domain_error("This activation functions is not implemented.");
	} // end switch

	return a;
} // end activation

/* -------------------------------------------------- 
forwardPass

Parameters: 
	x - vector to apply activation func to
	  - can be a matrix in Row-Major form 

Puts input x through the network and generates a prediction
If x is a matrix where each column is one input, it will do all as a batch
Stores each layer's unactivated value and activated one too for output layer

Returns:
	out - vector with the intermediate values and output 
		- if network is n-m-k and batch_size is 5, then out will have
		5m + 5k + 5k elements, 5 because of the batchsize
-------------------------------------------------- */
std::vector<std::vector<double> > NeuralNet::forwardPass(std::vector<double> &x)
{
	// -- Error Check --------------------------------------------------------
	double tmp = (double)(x.size()) / (double)(layers[0]);

	if (tmp < 1.0)
	{
		char msg [100];
		std::sprintf(msg, "User Input Size: %lud ; NN Input Size: %d", x.size(), layers[0]);
		throw std::length_error(msg);
	} // end if
	
	if (std::floor(tmp) != tmp)
	{
		char msg [100];
		std::sprintf(msg, "Too many/few Input Args (in_size / nn_in_size = %f)", tmp);
		throw std::length_error(msg);
	} // end if
	// ----------------------------------------------------------------------

	int batch_size = x.size() / layers[0];  // num cols in x
	int input_size = layers[0];				// num rows in x
	std::vector<double> tmp_v(x);
	std::vector<std::vector<double> > out(layers.size() + 1); 	

	out[0] = x;

	for (int i = 1; i < layers.size(); i++)
	{
		out[i] = matMulGPU(weights[i - 1], tmp_v, layers[i], layers[i - 1], batch_size);
		// out[i] = z_(i + 1)
		// weights[i] is layers[i + 1] x layers[i]
		// tmp_v is layers[i] x batch_size
		// out[i] is layers[i + 1] x batch_size

		tmp_v = activation(out[i], funcs[i - 1]);	
		// tmp_v = a_(i + 1)

	} // end for

	*(out.end() - 1) = tmp_v;

	return out;
} // end forwardPass

/* -------------------------------------------------- 
calcLoss

Parameters: 
	x - vector of predicted outputs 
	  - can be a matrix in Row-Major form 
	y - vector of actual outputs
	  - can be a matrix in Row-Major form

Applies whatever loss function is specified by NeuralNet.errFunc

Returns:
	err - double which is average error for batch of inputs
-------------------------------------------------- */
double NeuralNet::calcLoss(std::vector<double>& x, std::vector<double>& y)
{
	// -- Error Check --------------------------------------------------------
	double tmp1 = (double)(x.size()) / (double)(layers[layers.size() - 1]);
	double tmp2 = (double)(y.size()) / (double)(layers[layers.size() - 1]);

	if (tmp1 < 1.0) 
	{
		char msg [100];
		std::sprintf(msg, "User Output Size: %lud ; NN Output Size: %d", x.size(), layers[layers.size() - 1]);
		throw std::length_error(msg);
	} // end if
	
	if (tmp2 < 1.0) 
	{
		char msg [100];
		std::sprintf(msg, "User Output Size: %lud ; NN Output Size: %d", y.size(), layers[layers.size() - 1]);
		throw std::length_error(msg);
	} // end if

	if (std::floor(tmp1) != tmp1)
	{
		char msg [100];
		std::sprintf(msg, "Too many/few Input Args (out_size / nn_out_size = %f)", tmp1);
		throw std::length_error(msg);
	} // end if

	if (std::floor(tmp2) != tmp2)
	{
		char msg [100];
		std::sprintf(msg, "Too many/few Input Args (out_size / nn_out_size = %f)", tmp2);
		throw std::length_error(msg);
	} // end if
	// ----------------------------------------------------------------------

	double err;

	switch (errFunc)
	{
		case mse:
			err = mseGPU(x, y, x.size() / layers[layers.size() - 1]);
			break;
		case logLoss:
			err = crossEntropyGPU(x, y, x.size() / layers[layers.size() - 1]);
			break;
		default:
			throw std::domain_error("This loss function has not been implemented");
	} // end switch

	return err;
} // end error

/* -------------------------------------------------- 
backwardPass

Parameters: 
	FP - result of forwardPass (should have z's for every layer and output value)
	y - actual output
	batch_size - samples in batch

Uses vector iterators to move back through the network and get the gradients for Cost wrt weights

Returns:
	dC - vector of vectors where each vector is the deriv of cost wrt some weight
-------------------------------------------------- */
std::vector<std::vector<double> > NeuralNet::backwardPass(std::vector<std::vector<double> > &FP, std::vector<double> &y, int batch_size)
{

	// -- Error Check -----------------------------------------------
	for (int i = 0; i < layers.size(); i++)
	{
		if (FP[i].size() / batch_size != layers[i])
			throw std::length_error("backwardPass: Invalid Vector size to FP");
		
	} // end for

	if (FP.back().size() / batch_size != layers.back())
		throw std::length_error("backwardsPass: Invalid Vector size to FP");
	
	if (y.size() / batch_size != layers.back())
		throw std::length_error("backwardPass: Invalid Vector size to y");

	if (batch_size < 1)
		throw std::invalid_argument("backwardPass: there must be at least one sample in batch");
	// -------------------------------------------------------------- 

	std::vector<std::vector<double> > dC(layers.size() - 1);
	std::vector<double> delta, a, w;

	std::vector<std::vector<double> >::iterator it_dC = dC.end() - 1, it_FP = FP.end() - 1, it_weights = weights.end() - 1;
	std::vector<int>::iterator it_layers = layers.end() - 1;
	std::vector<Activation>::iterator it_funcs = funcs.end() - 1;
	
	switch (errFunc)
	{
		case mse:
			delta = msePrimeGPU(*it_FP, y, *it_layers, batch_size);
			break;
		case logLoss:
			delta = crossEntropyPrimeGPU(*it_FP, y, *it_layers, batch_size);
			break;
		default:
			throw std::domain_error("backwardPass: This loss func hasn't been implemented");
	} // end switch

	// a <- act'(zL)
	a = activation(*(it_FP - 1), *it_funcs, true);

	// delta <- del_a(C) o act'(zL) = deltaL
	delta = hadamardGPU(delta, a, *it_layers, batch_size);

	// a <- a(z(L-1))
	a = activation(*(it_FP - 2), *(it_funcs - 1));

	// a <- act(z(L-1))^T
	a = matTransGPU(a, *(it_layers - 1), batch_size);

	// dC[l] <- (deltaL)(a[L-1])^T
	*it_dC = matMulGPU(delta, a, *it_layers, batch_size, *(it_layers - 1));

	// adjust gradients for batch_size
	*it_dC = scalarMultGPU(*it_dC, 1.0/(double)(batch_size), *it_layers, *(it_layers - 1));

	// pts to z(L-1)
	it_FP -= 2;

	// pts to layers[L - 1]
	it_layers -= 1;

	// pts to funcs[L - 1]
	it_funcs -= 1;

	// pts to it_dC[L - 1]
	it_dC -= 1;

	// it_weights already points to weights[L]

	while (it_dC != dC.begin())
	{
		// l denotes layer 

		// w <- w[l+1]^T    [l+1 x l --> l x l+1] 
		w = matTransGPU(*it_weights, *(it_layers + 1), *(it_layers));

		// delta <- (w[l+1]^T)(delta[l+1])
		delta = matMulGPU(w, delta, *it_layers, *(it_layers + 1), batch_size);

		// a <- act'(z[l])
		a = activation(*it_FP, *it_funcs, true);

		// delta <- delta[l]
		delta = hadamardGPU(delta, a, *it_layers, batch_size);

		// a <- act(z(l-1))
		a = activation(*(it_FP - 1), *(it_funcs - 1));

		// a <- act(z(l-1))^T
		a = matTransGPU(a, *(it_layers - 1), batch_size);

		// it_dC[l] <- (delta[l])(act[l-1]^T)
		*it_dC = matMulGPU(delta, a, *it_layers, batch_size, *(it_layers - 1));

		// adjust gradients for batch_size
		*it_dC = scalarMultGPU(*it_dC, 1.0/(double)(batch_size), *it_layers, *(it_layers - 1));

		// decrement iterators
		it_FP -= 1;
		it_layers -= 1;
		it_funcs -= 1;
		it_weights -= 1;
		it_dC -= 1;	
	} // end while

	// w <- w[l+1]^T    [l+1 x l --> l x l+1] 
	w = matTransGPU(*it_weights, *(it_layers + 1), *(it_layers));

	// delta <- (w[l+1]^T)(delta[l+1])
	delta = matMulGPU(w, delta, *it_layers, *(it_layers + 1), batch_size);

	// a <- act'(z2)
	a = activation(*it_FP, *it_funcs, true);

	// delta <- delta2
	delta = hadamardGPU(delta, a, *it_layers, batch_size);

	// a <- x
	a = *(it_FP - 1);

	// a <- x^T
	a = matTransGPU(a, *(it_layers - 1), batch_size);

	// it_dC[1] <- (delta[1])(x^T)
	*it_dC = matMulGPU(delta, a, *it_layers, batch_size, *(it_layers - 1));

	// adjust gradients for batch_size
	*it_dC = scalarMultGPU(*it_dC, 1.0/(double)(batch_size), *it_layers, *(it_layers - 1));

	return dC;
} // end backwardPass

/* ----------------------------------------------
updateWeights

Paramters:
	W - weight matrix as vector
	dC - dC/dW matrix as vector
	alpha - learning rate
	len - number of entries in W and dC

Assigns new value for each weight using alpha
---------------------------------------------- */
__global__ void updateWeights(double *W, double *dC, double alpha, int len)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < len)
	{
		W[idx] = W[idx] - alpha * dC[idx];
	} // end if

} // end updateWeights

/* ----------------------------------------------
sgdConstLR

Parameters:
	dC - gradient matrices passed to kernel
	alpha - learning rate

Updates using vanilla grad descent approach

Far better implementation is Adam
---------------------------------------------- */
void NeuralNet::sgdConstLR(std::vector<std::vector<double> >& dC, double alpha)
{
	for (int i = 0; i < dC.size(); i++)
	{
		double *d_W, *d_dC;
		int BLOCKSIZE = dC[i].size() >= 512 ? 512 : dC[i].size();

		cudaMalloc((void **) &d_W, weights[i].size() * sizeof(double));
		cudaMalloc((void **) &d_dC, dC[i].size() * sizeof(double));

		cudaMemcpy(d_W, weights[i].data(), weights[i].size() * sizeof(double), cudaMemcpyHostToDevice);	
		cudaMemcpy(d_dC, dC[i].data(), dC[i].size() * sizeof(double), cudaMemcpyHostToDevice);

		dim3 GRID((dC[i].size() + BLOCKSIZE - 1) / BLOCKSIZE);
		dim3 BLOCK(BLOCKSIZE);

		updateWeights<<<GRID, BLOCK, 0>>>(d_W, d_dC, alpha, dC[i].size());

		cudaMemcpy(weights[i].data(), d_W, weights[i].size() * sizeof(double), cudaMemcpyDeviceToHost);

		cudaFree(d_W);
		cudaFree(d_dC);
	} // end for

} // end updateWeightsGPU

/* ----------------------------------------------
updateWeightsVect

Parameters:
	W - matrix of weights to update
	dC - matrix of weight grads
	alpha - matrix of alphas
	len - length of each vect as matrix	

Sets new value for each weight using alpha and dC
---------------------------------------------- */
__global__ void updateWeightsVect(double *W, double *dC, double *alpha, int len)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < len)
	{
		W[idx] = W[idx] - alpha[idx] * dC[idx];
	} // end if

} // end updateWeightsVect

/* ----------------------------------------------
sgdAdam

Parameters:

s
---------------------------------------------- */
void NeuralNet::sgdADAM(std::vector<std::vector<double> >& dC, double alpha, double beta1, double beta2, double epsilon)
{
	double pwr;
	std::vector<double> tmp1, tmp2, gSqr, corrM, corrV, eps;

	for (int i = 0; i < dC.size(); i++)
	{
		gSqr = hadamardGPU(dC[i], dC[i], layers[i + 1], layers[i]);

		tmp1 = scalarMultGPU(gradWeightedMean[i], beta1, layers[i + 1], layers[i]);
		tmp2 = scalarMultGPU(dC[i], (1 - beta1), layers[i + 1], layers[i]);
		gradWeightedMean[i] = matAddGPU(tmp1, tmp2, layers[i + 1], layers[i]);
		
		pwr = std::pow(beta1, adamIters);
		corrM = scalarMultGPU(gradWeightedMean[i], 1.0 / (1 - pwr), layers[i + 1], layers[i]);

		tmp1 = scalarMultGPU(gradBiasedVariance[i], beta2, layers[i + 1], layers[i]);
		tmp2 = scalarMultGPU(gSqr, (1 - beta2), layers[i + 1], layers[i]);
		gradBiasedVariance[i] = matAddGPU(tmp1, tmp2, layers[i + 1], layers[i]);

		pwr = std::pow(beta2, adamIters);
		corrV = scalarMultGPU(gradBiasedVariance[i], 1.0 / (1 - pwr), layers[i + 1], layers[i]);

		// tmp1 becomes alpha/(sqrt(corrV) + eps)
		eps.insert(eps.begin(), corrV.size(), epsilon);
		tmp1 = matSqrtGPU(corrV, layers[i + 1], layers[i]);
		tmp1 = matAddGPU(tmp1, eps, layers[i + 1], layers[i]);
		tmp1 = matReciprocalGPU(tmp1, layers[i + 1], layers[i]);
		tmp1 = scalarMultGPU(tmp1, alpha, layers[i + 1], layers[i]);
		eps.clear();

		// calling updateWeightsVect kernel
		double *d_W, *d_dC, *d_alpha;
		int BLOCKSIZE = layers[i + 1] * layers[i] >= 512 ? 512 : layers[i + 1] * layers[i];

		cudaMalloc((void **) &d_W, layers[i + 1] * layers[i] * sizeof(double));
		cudaMalloc((void **) &d_dC, layers[i + 1] * layers[i] * sizeof(double));
		cudaMalloc((void **) &d_alpha, layers[i + 1] * layers[i] * sizeof(double));

		cudaMemcpy(d_W, weights[i].data(), layers[i + 1] * layers[i] * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_dC, corrM.data(), layers[i + 1] * layers[i] * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_alpha, tmp1.data(), layers[i + 1] * layers[i] * sizeof(double), cudaMemcpyHostToDevice);

		dim3 GRID((layers[i + 1] * layers[i] + BLOCKSIZE - 1) / BLOCKSIZE);
		dim3 BLOCK(BLOCKSIZE);

		updateWeightsVect<<<GRID, BLOCK, 0>>>(d_W, d_dC, d_alpha, layers[i + 1] * layers[i]);

		cudaMemcpy(weights[i].data(), d_W, layers[i + 1] * layers[i] * sizeof(double), cudaMemcpyDeviceToHost);

		cudaFree(d_W);
		cudaFree(d_dC);
		cudaFree(d_alpha);
	} // end for
	
	adamIters += 1;
} // end sgdADAM

/* ---------------------------------------------- 
train

Parameters:
	x - vector of batches of inputs
	y - vector of batches of outputs
	batch_size - number of samples per batch
	lr - determines algorithm to set learning rate
	alpha - only needs to be set if lr = constant
		  - determines learning rate for sgd

Applies a loop over FP, BP, and SGD for all batches passed
Result is neural net that has been trained using given examples
---------------------------------------------- */
void NeuralNet::train(std::vector<std::vector<double> >& x, std::vector<std::vector<double> >& y, int batch_size, double alpha)
{
	// --- Error Check -----------------------------
	if (x.size() != y.size())
		throw std::length_error("train: x and y have incompatible sizes");

	for (int i = 0; i < x.size(); i++)
	{
		if (x[i].size() / batch_size != layers[0])
			throw std::length_error("train: x has invalid input sizes");

	} // end for	

	for (int i = 0; i < y.size(); i++)
	{
		if (y[i].size() / batch_size != layers.back())
			throw std::length_error("train: y has invalid size");

	} // end for
	// ---------------------------------------------

	std::vector<std::vector<double> > FP, dC;

	for (int i = 0; i < x.size(); i++)
	{
		FP = forwardPass(x[i]);
		dC = backwardPass(FP, y[i], batch_size);

		switch (alphaAlg)
		{
			case constant:
				sgdConstLR(dC, alpha);
				break;
			case adam:
				sgdADAM(dC);
				break;
			default:
				throw std::domain_error("train: this lr optimizer isn't implemented");
		} // end switch

	} // end for

} // end train

/* -------------------------------------------------- 
printNN

Prints the size of each layer, the activation function
	at each layer, and the loss function at the end of the network
-------------------------------------------------- */
void NeuralNet::printNN() const
{
	std::cout << "Layer 0: " << layers[0] << std::endl;	

	for (int i = 1; i < layers.size(); i++)
	{
		std::cout << std::endl;
		std::cout << "Layer " << i << ": " << layers[i] << std::endl;
		
		switch (funcs[i - 1])
		{
			case binary_step: 
				std::cout << "Activation: Binary Step" << std::endl;
				break;
			case sigmoid: 
				std::cout << "Activation: Sigmoid" << std::endl;
				break;
			case relu: 
				std::cout << "Activation: ReLU" << std::endl;
				break;
			case leaky_relu: 
				std::cout << "Activation: Leaky ReLU" << std::endl;
				break;
			case exponential:
				std::cout << "Activation: Exponential" << std::endl;
				break;
			default:
				throw std::domain_error("This activation function is not implemented");
		} // end switch

	} // end for

	std::cout << std::endl;

	switch(errFunc)
	{
		case mse:
			std::cout << "Loss Function: Mean-Squared Error" << std::endl;
			break;
		case logLoss:
			std::cout << "Loss Function: Cross Entropy" << std::endl;
			break;
		default:
			throw std::domain_error("This loss function is not implemented");
	} // end switch

	std::cout << std::endl;

	switch (alphaAlg)
	{
		case constant:
			std::cout << "LR Optimizer: None" << std::endl;
			break;
		case adam:
			std::cout << "LR Optimizer: Adam" << std::endl;
			break;
		default:
			throw std::domain_error("printNN: Alpha optimizer not implemented");
	} // end swithc

} // end printNN

/* -------------------------------------------------- 
printWeights

Parameter:
	l - layers to print weights for (range is 0 to layers.size() - 2)

Prints all the weights from layer l to l + 1
-------------------------------------------------- */
void NeuralNet::printWeights(int l) const 
{
	if (l < 0)
		throw std::length_error("Not a layer");

	if (l > layers.size() - 2)
		throw std::domain_error("There are no weights for this layer");

	std::cout << "Weights for layers " << l << " to " << l + 1 << ":" << std::endl;
	
	for (int i = 0; i < layers[l + 1]; i++) 
	{	
		for (int j = 0; j < layers[l]; j++) 
			std::cout << weights[l][i * layers[l] + j] << "  ";			
		
		std::cout << std::endl;

	} // end for

	std::cout << std::endl;

} // end printWeights

