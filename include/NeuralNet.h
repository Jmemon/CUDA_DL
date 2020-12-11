#ifndef NEURALNET_H
#define NEURALNET_H

typedef enum Activation {
	binary_step,
	sigmoid,
	// tanh,
	relu,
	leaky_relu
} Activation ;

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

#endif	// NEURALNET_H
