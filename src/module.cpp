#include "../include/module.h"
#include <algorithm>

std::vector<double> Module::get_parameter_group(const std::string& name) const {
    for (const auto& group : param_groups) {
        if (group.name == name) {
            return std::vector<double>(parameters.begin() + group.offset, 
                                     parameters.begin() + group.offset + group.size);
        }
    }
    return std::vector<double>();
}

std::vector<double> Module::get_parameter_gradient_group(const std::string& name) const {
    for (const auto& group : param_groups) {
        if (group.name == name) {
            return std::vector<double>(parameter_gradients.begin() + group.offset,
                                     parameter_gradients.begin() + group.offset + group.size);
        }
    }
    return std::vector<double>();
}

void Module::add_parameter_group(const std::string& name, int size) {
    int offset = parameters.size();
    param_groups.push_back({offset, size, name});
    parameters.resize(parameters.size() + size);
    parameter_gradients.resize(parameter_gradients.size() + size);
}