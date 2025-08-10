#ifndef LINEAR_H
#define LINEAR_H

#include "module.h"

class Linear : public Module {
private:
    int in_size;
    int out_size;

public:
    Linear(int in_size, int out_size);
    
    std::vector<double> forward(std::vector<double> &x) override;
    std::vector<double> backward(std::vector<double> &dL_dz) override;
    
    std::vector<double> get_weights() const;
    std::vector<double> get_bias() const;
    std::vector<double> get_weight_gradients() const;
    std::vector<double> get_bias_gradients() const;
    
private:
    void initialize_parameters();
};

#endif