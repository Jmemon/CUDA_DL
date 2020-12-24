#include "../include/NeuralNet.h"
#include "../include/Activation.cuh"
#include <iostream>
#include <vector>

int main(int argc, char *argv[]) {
	
	std::vector<int> layers(3, 0);
	layers[0] = 100;
	layers[1] = 500;
	layers[2] = 10;

	std::vector<Activation> funcs(3, relu);
	funcs[0] = binary_step;
	funcs[1] = relu;
	funcs[2] = relu;

	NeuralNet nn(layers, funcs);

	nn.printWeights(0);
	std::cout << std::endl;

	nn.printWeights(1);
	std::cout << std::endl;	

	return 0;
}
