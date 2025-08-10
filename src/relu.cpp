#include "../include/module.h"
#include <algorithm>

class ReLU : public Module {
public:
    ReLU() = default;
    
    std::vector<double> forward(std::vector<double> &x) override {
        input_buffer = x;
        std::vector<double> output(x.size());
        
        for (size_t i = 0; i < x.size(); ++i) {
            output[i] = std::max(0.0, x[i]);
        }
        
        return output;
    }
    
    std::vector<double> backward(std::vector<double> &dL_dz) override {
        input_gradients.resize(input_buffer.size());
        
        for (size_t i = 0; i < input_buffer.size(); ++i) {
            input_gradients[i] = (input_buffer[i] > 0.0) ? dL_dz[i] : 0.0;
        }
        
        return input_gradients;
    }
};