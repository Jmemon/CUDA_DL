#ifndef PARALLELNN_CUH
#define PARALLELNN_CUH

typedef enum Activation {
	binary_step,
	sigmoid,
	// tanh,
	relu,
	leaky_relu
} Activation ;

extern __global__ void randInit(double *w);

extern __device__ double max(double a, double b);

extern __global__ void binary_step(double *x);

extern __global__ void sigmoid(double *x);

extern __global__ void relu(double *x);

extern __global__ void leaky_relu(double *x);

#endif //PARALLELNN_CUH
