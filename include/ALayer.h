#ifndef ALAYER_H
#define ALAYER_H

#include "Layer.h"
#include "NeuralNet.h"

#include <vector>
#include <deque>

class ALayer: public Layer {

	public:
		// Default Constructor: sets act to relu
		ALayer();

		// Constructor: Sets act to act
		ALayer(int layerSize, Activation act);

		// Input: the unactivated vector
		// Output: the activated vector
		// Calls the specified activation function on in
		// Also appends in to end of inputs
		std::vector<double> forwardPropagation(std::vector<double> in);

		// Input: vector with derivatives of cost wrt to each activated value
		// Output: vector with derivatives of cost wrt to each unactivated value
		// Calculates: act'(inputs[0]) Hadamaard delta_out
		// Also removes inputs[0] from inputs
		std::vector<double> backPropagation(std::vector<double> delta);

	private:
		int layerSize;
		Activation act;
		std::deque<std::vector<double>> inputs;
} 

#endif
