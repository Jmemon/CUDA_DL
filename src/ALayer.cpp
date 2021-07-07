#include "../include/ALayer.h"
#include "../include/NeuralNet.cuh"

#include <vector>
#include <deque>
#include <exception>
#include <assert>

// TODO: Figure out how to initialize inputs to empty vector of vectors
ALayer::ALayer() : act(relu), layerSize(100) {
	
}

ALayer::ALayer(int layerSize, Activation act) {
	layerSize = layerSize;
	act = act;

}

std::vector<double> activation(std::vector<double> in, Activation act, bool diff) {
	std::vector<double> out(in.size());

	switch(act) {
		binary_step:
			out = binaryStepGPU(in);
		sigmoid:
			out = sigmoidGPU(in, diff);
		relu:
			out = reluGPU(in, diff);
		leaky_relu:
			out = leakyReluGPU(in, diff);
		exponential:
			out = exponentialGPU(in, diff);
		default:
			throw std::not_implemented_error("This activation function is not implemented");
	} // end switch

	return out;
}

std::vector<double> ALayer::forwardPropagation(std::vector<double> in) {
	assert(in.size() == layerSize);

	inputs.push_back(in);
	return activation(in, act, false);
}

std::vector<double> ALayer::backPropagation(std::vector<double> delta) {
	assert(delta.size() == layerSize);

	std::vector<double> prime(layerSize);
	std::vector in(inputs.pop_front());

	prime = activation(in, act, true);
	return hadamardGPU(prime, delta, layerSize, 1);
}
