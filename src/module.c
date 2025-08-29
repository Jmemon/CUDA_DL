#include "../include/module.h"
#include <stdlib.h>
#include <string.h>

#define HASHMAP_DEFAULT_SIZE 16

// Module creation and management
Module* module_create(const char* name, char** param_names, int num_params, 
                     char** hyperparam_names, int num_hyperparams) {
    Module* module = (Module*)malloc(sizeof(Module));
    if (!module) return NULL;
    
    // Initialize name
    if (name) {
        module->name = (char*)malloc((strlen(name) + 1) * sizeof(char));
        strcpy(module->name, name);
    } else {
        module->name = NULL;
    }
    
    // Initialize parameter names
    module->num_param_names = num_params;
    if (num_params > 0) {
        module->param_names = (char**)malloc(num_params * sizeof(char*));
        for (int i = 0; i < num_params; i++) {
            module->param_names[i] = (char*)malloc((strlen(param_names[i]) + 1) * sizeof(char));
            strcpy(module->param_names[i], param_names[i]);
        }
    } else {
        module->param_names = NULL;
    }
    
    // Initialize hyperparameter names
    module->num_hyperparam_names = num_hyperparams;
    if (num_hyperparams > 0) {
        module->hyperparam_names = (char**)malloc(num_hyperparams * sizeof(char*));
        for (int i = 0; i < num_hyperparams; i++) {
            module->hyperparam_names[i] = (char*)malloc((strlen(hyperparam_names[i]) + 1) * sizeof(char));
            strcpy(module->hyperparam_names[i], hyperparam_names[i]);
        }
    } else {
        module->hyperparam_names = NULL;
    }
    
    // Initialize hashmaps
    module->param_values = hashmap_create(HASHMAP_DEFAULT_SIZE);
    module->hyperparam_values = hashmap_create(HASHMAP_DEFAULT_SIZE);
    module->param_grad_buffers = hashmap_create(HASHMAP_DEFAULT_SIZE);
    module->input_buffers = hashmap_create(HASHMAP_DEFAULT_SIZE);
    module->input_grad_buffers = hashmap_create(HASHMAP_DEFAULT_SIZE);
    module->moment_m = hashmap_create(HASHMAP_DEFAULT_SIZE);
    module->moment_v = hashmap_create(HASHMAP_DEFAULT_SIZE);
    
    // Initialize function pointers to NULL
    module->forward = NULL;
    module->backward = NULL;
    
    return module;
}

void module_set_functions(Module* module, module_forward_fn forward, 
                         module_backward_fn backward) {
    if (!module) return;
    
    module->forward = forward;
    module->backward = backward;
}

// Helper function to free DoubleTensor pointers
static void free_tensor_ptr(void* tensor_ptr) {
    DoubleTensor* tensor = (DoubleTensor*)tensor_ptr;
    if (tensor) {
        double_tensor_free(tensor);
        free(tensor);
    }
}

// Helper function to free double pointers
static void free_double_ptr(void* double_ptr) {
    double* d = (double*)double_ptr;
    if (d) {
        free(d);
    }
}

void module_free(Module* module) {
    if (!module) return;
    
    // Free name
    if (module->name) {
        free(module->name);
    }
    
    // Free parameter names
    if (module->param_names) {
        for (int i = 0; i < module->num_param_names; i++) {
            if (module->param_names[i]) {
                free(module->param_names[i]);
            }
        }
        free(module->param_names);
    }
    
    // Free hyperparameter names
    if (module->hyperparam_names) {
        for (int i = 0; i < module->num_hyperparam_names; i++) {
            if (module->hyperparam_names[i]) {
                free(module->hyperparam_names[i]);
            }
        }
        free(module->hyperparam_names);
    }
    
    // Free hashmaps
    hashmap_free(module->param_values, free_tensor_ptr);
    hashmap_free(module->hyperparam_values, free_double_ptr);
    hashmap_free(module->param_grad_buffers, free_tensor_ptr);
    hashmap_free(module->input_buffers, free_tensor_ptr);
    hashmap_free(module->input_grad_buffers, free_tensor_ptr);
    hashmap_free(module->moment_m, free_tensor_ptr);
    hashmap_free(module->moment_v, free_tensor_ptr);
    
    free(module);
}

// Parameter validation and access functions
int module_validate_param_name(Module* module, const char* param_name) {
    if (!module || !param_name) return 0;
    
    for (int i = 0; i < module->num_param_names; i++) {
        if (strcmp(module->param_names[i], param_name) == 0) {
            return 1;
        }
    }
    return 0;
}

DoubleTensor* module_get_param_value(Module* module, const char* param_name) {
    if (!module || !param_name || !module_validate_param_name(module, param_name)) {
        return NULL;
    }
    
    return (DoubleTensor*)hashmap_get(module->param_values, param_name);
}

void module_set_param_value(Module* module, const char* param_name, DoubleTensor* tensor) {
    if (!module || !param_name || !module_validate_param_name(module, param_name)) {
        return;
    }
    
    hashmap_put(module->param_values, param_name, tensor);
}

double module_get_hyperparam_value(Module* module, const char* hyperparam_name) {
    if (!module || !hyperparam_name) return 0.0;
    
    double* value = (double*)hashmap_get(module->hyperparam_values, hyperparam_name);
    return value ? *value : 0.0;
}

void module_set_hyperparam_value(Module* module, const char* hyperparam_name, double value) {
    if (!module || !hyperparam_name) return;
    
    double* value_ptr = (double*)malloc(sizeof(double));
    *value_ptr = value;
    hashmap_put(module->hyperparam_values, hyperparam_name, value_ptr);
}

// Buffer management
DoubleTensor* module_get_param_grad(Module* module, const char* param_name) {
    if (!module || !param_name || !module_validate_param_name(module, param_name)) {
        return NULL;
    }
    
    return (DoubleTensor*)hashmap_get(module->param_grad_buffers, param_name);
}

void module_set_param_grad(Module* module, const char* param_name, DoubleTensor* grad) {
    if (!module || !param_name || !module_validate_param_name(module, param_name)) {
        return;
    }
    
    hashmap_put(module->param_grad_buffers, param_name, grad);
}

DoubleTensor* module_get_input_buffer(Module* module, const char* buffer_name) {
    if (!module || !buffer_name) return NULL;
    
    return (DoubleTensor*)hashmap_get(module->input_buffers, buffer_name);
}

void module_set_input_buffer(Module* module, const char* buffer_name, DoubleTensor* buffer) {
    if (!module || !buffer_name) return;
    
    hashmap_put(module->input_buffers, buffer_name, buffer);
}

DoubleTensor* module_get_input_grad_buffer(Module* module, const char* buffer_name) {
    if (!module || !buffer_name) return NULL;
    
    return (DoubleTensor*)hashmap_get(module->input_grad_buffers, buffer_name);
}

void module_set_input_grad_buffer(Module* module, const char* buffer_name, DoubleTensor* buffer) {
    if (!module || !buffer_name) return;
    
    hashmap_put(module->input_grad_buffers, buffer_name, buffer);
}

// Adam optimizer moment estimates
DoubleTensor* module_get_moment_m(Module* module, const char* param_name) {
    if (!module || !param_name || !module_validate_param_name(module, param_name)) {
        return NULL;
    }
    
    return (DoubleTensor*)hashmap_get(module->moment_m, param_name);
}

void module_set_moment_m(Module* module, const char* param_name, DoubleTensor* m) {
    if (!module || !param_name || !module_validate_param_name(module, param_name)) {
        return;
    }
    
    hashmap_put(module->moment_m, param_name, m);
}

DoubleTensor* module_get_moment_v(Module* module, const char* param_name) {
    if (!module || !param_name || !module_validate_param_name(module, param_name)) {
        return NULL;
    }
    
    return (DoubleTensor*)hashmap_get(module->moment_v, param_name);
}

void module_set_moment_v(Module* module, const char* param_name, DoubleTensor* v) {
    if (!module || !param_name || !module_validate_param_name(module, param_name)) {
        return;
    }
    
    hashmap_put(module->moment_v, param_name, v);
}

void module_init_optimizer_state(Module* module) {
    if (!module) return;
    
    // Initialize moment estimates for all parameters
    for (int i = 0; i < module->num_param_names; i++) {
        const char* param_name = module->param_names[i];
        DoubleTensor* param = module_get_param_value(module, param_name);
        
        if (param) {
            // Initialize first moment estimate
            DoubleTensor* m = (DoubleTensor*)malloc(sizeof(DoubleTensor));
            *m = double_tensor_alloc(param->dims, param->num_dims);
            double_tensor_zero_init(m);
            module_set_moment_m(module, param_name, m);
            
            // Initialize second moment estimate
            DoubleTensor* v = (DoubleTensor*)malloc(sizeof(DoubleTensor));
            *v = double_tensor_alloc(param->dims, param->num_dims);
            double_tensor_zero_init(v);
            module_set_moment_v(module, param_name, v);
        }
    }
}

// Utility functions
void module_zero_gradients(Module* module) {
    if (!module) return;
    
    for (int i = 0; i < module->num_param_names; i++) {
        const char* param_name = module->param_names[i];
        DoubleTensor* grad = module_get_param_grad(module, param_name);
        
        if (grad) {
            double_tensor_zero_init(grad);
        }
    }
}

void module_get_all_parameters(Module* module, char*** param_names, DoubleTensor*** param_values, int* total_params) {
    if (!module || !param_names || !param_values || !total_params) return;
    
    *total_params = module->num_param_names;
    
    if (*total_params == 0) {
        *param_names = NULL;
        *param_values = NULL;
        return;
    }
    
    *param_names = (char**)malloc(*total_params * sizeof(char*));
    *param_values = (DoubleTensor**)malloc(*total_params * sizeof(DoubleTensor*));
    
    for (int i = 0; i < *total_params; i++) {
        (*param_names)[i] = module->param_names[i];
        (*param_values)[i] = module_get_param_value(module, module->param_names[i]);
    }
}

// Module type validation functions
int module_validate_linear(Module* module) {
    if (!module) return 0;
    
    // Check if module has required parameters for linear layer
    return module_validate_param_name(module, "weight") &&
           module_validate_param_name(module, "bias");
}

int module_validate_conv2d(Module* module) {
    if (!module) return 0;
    
    // Check if module has required parameters for conv2d layer
    return module_validate_param_name(module, "kernel");
}

int module_validate_relu(Module* module) {
    if (!module) return 0;
    
    // ReLU has no parameters, so any module is valid for ReLU
    return 1;
}