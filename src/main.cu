#include "../include/NeuralNet.h"
#include "../include/Activation.cuh"
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "../include/Loss.cuh"

void randVect(std::vector<double> &x);

double normVect(std::vector<double> &x);

std::ostream& operator<< (std::ostream& cout, std::vector<double> &x);

void operator/= (std::vector<double> &x, double c);

int main(int argc, char *argv[]) {

	std::vector<double> x(500);
	// batch_size = 5

	std::vector<double> y(10, 1);

	std::vector<double> err(5);

	randVect(x);
	x /= normVect(x);

	std::vector<int> layers(3);
	layers[0] = 100;
	layers[1] = 50;
	layers[2] = 2;

	std::vector<Activation> funcs(2);
	funcs[0] = sigmoid;
	funcs[1] = leaky_relu;

	NeuralNet nn(layers, funcs);

	nn.printNN();
	nn.forwardPass(x);

	err = mseGPU(x.data(), y.data(), 2, 5);

	std::cout << "x: " << x;
	std::cout << "y: " << y;
	std::cout << "err: " << err;

	return 0;
}

void randVect(std::vector<double> &x)
{
	srand(time(NULL));

	for (int i = 0; i < x.size(); i++)
		x[i] = (double)(rand() % 1000000) / 1000.0;

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
