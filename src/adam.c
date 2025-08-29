#include "../include/adam.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

Adam adam_create(float lr, float beta1, float beta2, float eps, Module* model) {
    Adam adam;
    adam.lr = lr;
    adam.beta1 = beta1;
    adam.beta2 = beta2;
    adam.eps = eps;
    adam.t = 0;
    adam.model = model;
    
    adam.name = (char*)malloc(5 * sizeof(char));
    strcpy(adam.name, "adam");
    
    // Initialize optimizer state for the entire model
    if (model) {
        module_init_optimizer_state(model);
    }
    
    return adam;
}

void adam_step(Adam* adam) {
    if (!adam || !adam->model) {
        return;
    }
    
    adam->t++;  // Increment time step
    
    // Get all parameters from the model
    char** param_names;
    DoubleTensor** param_values;
    int total_params;
    
    module_get_all_parameters(adam->model, &param_names, &param_values, &total_params);
    
    // Bias correction terms
    float lr_corrected = adam->lr * sqrt(1.0f - pow(adam->beta2, adam->t)) / (1.0f - pow(adam->beta1, adam->t));
    
    // Update all parameters
    for (int i = 0; i < total_params; i++) {
        const char* param_name = param_names[i];
        DoubleTensor* param = param_values[i];
        
        // Get gradient and moment estimates
        DoubleTensor* grad = module_get_param_grad(adam->model, param_name);
        DoubleTensor* m = module_get_moment_m(adam->model, param_name);
        DoubleTensor* v = module_get_moment_v(adam->model, param_name);
        
        // Skip parameters that don't have necessary components
        if (!param || !grad || !m || !v) {
            continue;
        }
        
        // Update each element of the parameter tensor
        for (int j = 0; j < param->num_elem; j++) {
            double grad_val = grad->elems[j];
            
            // Update biased first moment estimate
            m->elems[j] = adam->beta1 * m->elems[j] + (1.0f - adam->beta1) * grad_val;
            
            // Update biased second raw moment estimate  
            v->elems[j] = adam->beta2 * v->elems[j] + (1.0f - adam->beta2) * grad_val * grad_val;
            
            // Update parameters with Adam rule
            param->elems[j] -= lr_corrected * m->elems[j] / 
                              (sqrt(v->elems[j]) + adam->eps);
        }
    }
    
    // Free the arrays allocated by module_get_all_parameters
    if (param_names) {
        free(param_names);
    }
    if (param_values) {
        free(param_values);
    }
    
    // Zero all gradients after parameter update
    module_zero_gradients(adam->model);
}

void adam_free(Adam* adam) {
    if (adam->name) {
        free(adam->name);
        adam->name = NULL;
    }
}