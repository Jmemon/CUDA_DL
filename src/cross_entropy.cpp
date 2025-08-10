#include "../include/module.h"
#include <cmath>
#include <algorithm>

class CrossEntropyLoss : public Module {
private:
    int num_classes;
    
public:
    CrossEntropyLoss(int num_classes) : num_classes(num_classes) {}
    
    std::vector<double> forward(std::vector<double> &x) override {
        input_buffer = x;
        int batch_size = x.size() / num_classes;
        std::vector<double> output(batch_size);
        
        for (int b = 0; b < batch_size; ++b) {
            double max_val = *std::max_element(x.begin() + b * num_classes, 
                                             x.begin() + (b + 1) * num_classes);
            
            double sum_exp = 0.0;
            for (int i = 0; i < num_classes; ++i) {
                sum_exp += std::exp(x[b * num_classes + i] - max_val);
            }
            
            double log_sum_exp = max_val + std::log(sum_exp);
            output[b] = -x[b * num_classes] + log_sum_exp;
        }
        
        return output;
    }
    
    std::vector<double> backward(std::vector<double> &dL_dz) override {
        int batch_size = input_buffer.size() / num_classes;
        input_gradients.resize(input_buffer.size());
        
        for (int b = 0; b < batch_size; ++b) {
            double max_val = *std::max_element(input_buffer.begin() + b * num_classes,
                                             input_buffer.begin() + (b + 1) * num_classes);
            
            double sum_exp = 0.0;
            for (int i = 0; i < num_classes; ++i) {
                sum_exp += std::exp(input_buffer[b * num_classes + i] - max_val);
            }
            
            for (int i = 0; i < num_classes; ++i) {
                double softmax = std::exp(input_buffer[b * num_classes + i] - max_val) / sum_exp;
                input_gradients[b * num_classes + i] = dL_dz[b] * (softmax - (i == 0 ? 1.0 : 0.0));
            }
        }
        
        return input_gradients;
    }
};