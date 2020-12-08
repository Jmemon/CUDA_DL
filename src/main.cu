#include "../include/NeuralNet.h"
#include "../include/ParallelNN.h"
#include <stdlib.h>

int main(int argc, char *argv[]) {
	
	int num_layers = 3;
	
	int* layers = (int *)malloc(num_layers * sizeof(int));
	layers[0] = 100;
	layers[1] = 500;
	layers[2] = 10;

	Activation* funcs = (Activation *)malloc(num_layers * sizeof(Activation));
	funcs[0] = sigmoid;
	funcs[1] = sigmoid;
	funcs[2] = relu;

	NeuralNet nn = NeuralNet(layers, funcs, num_layers);

	return 0;
}
