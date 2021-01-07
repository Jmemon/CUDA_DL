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

	std::vector<int> layers;
	layers.push_back(500);
	layers.push_back(100);
	layers.push_back(10);
	layers.push_back(1);

	std::vector<Activation> funcs;
	funcs.push_back(sigmoid);
	funcs.push_back(leaky_relu);
	funcs.push_back(relu);

	Loss lFunc = logLoss;
	LROptim lr = constant;

	NeuralNet nn(layers, funcs, lFunc, lr);
	nn.printNN();

	int samples = 50;
	int batch_size = 50;
	int input_neurons = 500;
	int output_neurons = 1;

	std::vector<std::vector<double> > x(samples), x_test(10);
	for (int i = 0; i < x.size(); i++)
	{
		std::vector<double> tmp(batch_size * input_neurons);
		x[i] = tmp;
		randVect(x[i]);

		if (i < 10)
		{
			std::vector<double> tmp1(batch_size * input_neurons);
			x_test[i] = tmp1;
			randVect(x_test[i]);
		} // end if 

	} // end for

	std::vector<std::vector<double> > y(samples), y_test(10);
	for (int i = 0; i < y.size(); i++)
	{
		std::vector<double> tmp(batch_size * output_neurons);
		y[i] = tmp;
		randVect(y[i]);

		if (i < 10)
		{
			std::vector<double> tmp1(batch_size * output_neurons);
			y_test[i] = tmp1;
			randVect(y_test[i]);
		} // end if 

	} // end for

	//double err = nn.calcLoss(x[0], y[0]);
	//std::cout << "err: " << err << std::endl;

	nn.printWeights(2);

	nn.train(x, y, batch_size);

	nn.printWeights(2);

	//err = nn.calcLoss(x[0], y[0]);
	//std::cout << "err: " << err << std::endl;

	/*
	FP = nn.forwardPass(x);

	std::cout << "pred: " << *(FP.end() - 1);
	std::cout << "actl: " << y << std::endl;

	std::vector<std::vector<double> > dC;
	dC = nn.backwardPass(FP, y, x.size() / layers[0]);

	nn.printWeights(2);
	
	nn.sgdADAM(dC);

	nn.printWeights(2);
	*/
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
