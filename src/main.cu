#include "../include/NeuralNet.h"
#include "../include/Activation.cuh"
#include <vector>


int main(int argc, char *argv[]) {
	
	vector<int> layers(3, 0);
	layers[0] = 100;
	layers[1] = 500;
	layers[2] = 10;

	vector<Activation> funcs(3, relu);
	funcs[0] = binary_step;
	funcs[1] = relu;
	funcs[2] = relu;

	NeuralNet nn(layers, funcs);

	return 0;
}
