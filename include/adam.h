#ifndef ADAM_H
#define ADAM_H

#include "module.h"

typedef struct {
    float lr;
    float beta1;
    float beta2;
    float eps;
    int t;              // Time step for bias correction
    Module* model;      // Pointer to the model being optimized
    char* name;
} Adam;

Adam adam_create(float lr, float beta1, float beta2, float eps, Module* model);
void adam_step(Adam* adam);
void adam_free(Adam* adam);

#endif