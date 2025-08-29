#ifndef MODULE_H
#define MODULE_H

#include "tensor.h"
#include "hashmap.h"

// Forward declarations
typedef struct Module Module;

// Function pointers for module operations
typedef void (*module_forward_fn)(Module*, DoubleTensor*, DoubleTensor*);
typedef void (*module_backward_fn)(Module*, DoubleTensor*, DoubleTensor*);

// Module structure with hashmaps as specified in the spec
struct Module {
    char* name;
    
    // Parameter names and values as hashmaps
    char** param_names;
    int num_param_names;
    HashMap* param_values;           // Map<string, DoubleTensor*>
    
    // Hyperparameter names and values as hashmaps  
    char** hyperparam_names;
    int num_hyperparam_names;
    HashMap* hyperparam_values;      // Map<string, double>
    
    // Gradient and buffer maps for backprop and weight updates
    HashMap* param_grad_buffers;     // Map<string, DoubleTensor*>
    HashMap* input_buffers;          // Map<string, DoubleTensor*>
    HashMap* input_grad_buffers;     // Map<string, DoubleTensor*>
    
    // Adam optimizer moment estimates
    HashMap* moment_m;               // Map<string, DoubleTensor*> - first moment
    HashMap* moment_v;               // Map<string, DoubleTensor*> - second moment
    
    // Function pointers for module-specific operations
    module_forward_fn forward;
    module_backward_fn backward;
};

// Module creation and management
Module* module_create(const char* name, char** param_names, int num_params, 
                     char** hyperparam_names, int num_hyperparams);
void module_set_functions(Module* module, module_forward_fn forward, 
                         module_backward_fn backward);
void module_free(Module* module);

// Parameter validation and access functions
int module_validate_param_name(Module* module, const char* param_name);
DoubleTensor* module_get_param_value(Module* module, const char* param_name);
void module_set_param_value(Module* module, const char* param_name, DoubleTensor* tensor);
double module_get_hyperparam_value(Module* module, const char* hyperparam_name);
void module_set_hyperparam_value(Module* module, const char* hyperparam_name, double value);

// Buffer management
DoubleTensor* module_get_param_grad(Module* module, const char* param_name);
void module_set_param_grad(Module* module, const char* param_name, DoubleTensor* grad);
DoubleTensor* module_get_input_buffer(Module* module, const char* buffer_name);
void module_set_input_buffer(Module* module, const char* buffer_name, DoubleTensor* buffer);
DoubleTensor* module_get_input_grad_buffer(Module* module, const char* buffer_name);
void module_set_input_grad_buffer(Module* module, const char* buffer_name, DoubleTensor* buffer);

// Adam optimizer moment estimates
DoubleTensor* module_get_moment_m(Module* module, const char* param_name);
void module_set_moment_m(Module* module, const char* param_name, DoubleTensor* m);
DoubleTensor* module_get_moment_v(Module* module, const char* param_name);
void module_set_moment_v(Module* module, const char* param_name, DoubleTensor* v);
void module_init_optimizer_state(Module* module);

// Utility functions
void module_zero_gradients(Module* module);
void module_get_all_parameters(Module* module, char*** param_names, DoubleTensor*** param_values, int* total_params);

// Module type validation functions
int module_validate_linear(Module* module);
int module_validate_conv2d(Module* module);
int module_validate_relu(Module* module);

#endif