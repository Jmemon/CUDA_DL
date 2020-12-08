#ifndef NEURALNET_CUH
#define NEURALNET_CUH

#include "ParallelNN.h"

class NeuralNet {
	private:
		int num_layers;
		int *layers;
		double **weights;
		Activation *funcs;
	public:
		NeuralNet(int *l, Activation *f, int n);	
		void initWeights();
		void printWeights(double* w, int l1, int l2) const;		
		void activation(double* x, int len, Activation f);
};

#endif	// NEURALNET_CUH
