#include "include/NeuralNet.h"
#include "include/Activation.cuh"
#include "include/Loss.cuh"
#include <iostream>
#include <stdlib.h>
#include <vector>

void randVect(std::vector<double> &x);

double normVect(std::vector<double> &x);

std::ostream& operator<< (std::ostream& cout, std::vector<double> &x);

void operator/= (std::vector<double> &x, double c);

int main(int argc, char *argv[]) {

	// batch_size = 10

	std::vector<double> x(5000);
	randVect(x);
	x /= normVect(x);

	std::vector<double> y(10, 310), tmp;

	std::vector<std::vector<double> > FP;

	std::vector<int> layers;
	std::vector<Activation> funcs;

	layers.push_back(500);
	layers.push_back(100);
	layers.push_back(10);
	layers.push_back(1);

	funcs.push_back(sigmoid);
	funcs.push_back(leaky_relu);
	funcs.push_back(relu);

	Loss lFunc = logLoss;

	LROptim lr = adam;

	NeuralNet nn(layers, funcs, lFunc, lr);

	nn.printNN();
	FP = nn.forwardPass(x);

	std::cout << "pred: " << *(FP.end() - 1);
	std::cout << "actl: " << y << std::endl;

	std::vector<std::vector<double> > dC;
	dC = nn.backwardPass(FP, y, x.size() / layers[0]);

	nn.printWeights(2);
	
	nn.sgdADAM(dC);

	nn.printWeights(2);

	return 0;
}

void randVect(std::vector<double> &x)
{
	srand(time(NULL));

	for (int i = 0; i < x.size(); i++)
		x[i] = (double)(rand() % 1000000) / 1000000.0;

} // end randVect

double normVect(std::vector<double> &x)
{
	double sqrSum = 0.0;

	for (int i = 0; i < x.size(); i++)
		sqrSum += x[i] * x[i];

	return std::sqrt(sqrSum);

} // end normVect

std::ostream& operator<< (std::ostream& cout, std::vector<double> &x)
{
	for (int i = 0; i < x.size(); i++)
		std::cout << x[i] << "  ";	

	std::cout << std::endl;

	return cout;
} // end operator<<

void operator/= (std::vector<double> &x, double c)
{
	for (int i = 0; i < x.size(); i++)
		x[i] /= c;

} // end operator/=
