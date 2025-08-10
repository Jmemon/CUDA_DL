#ifndef MODULE_H
#define MODULE_H

#include <vector>
#include <string>

struct ParameterGroup {
    int offset;
    int size;  // number of parameters (doubles) in this group
    std::string name;
};

class Module {
public:
    std::vector<double> parameters;
    std::vector<double> input_buffer;
    std::vector<double> parameter_gradients;
    std::vector<double> input_gradients;

protected:
    std::vector<ParameterGroup> param_groups;

public:
    Module() = default;
    virtual ~Module() = default;

    std::vector<double> get_parameter_group(const std::string& name) const;
    std::vector<double> get_parameter_gradient_group(const std::string& name) const;
    
    virtual std::vector<double> forward(std::vector<double> &x) = 0;
    virtual std::vector<double> backward(std::vector<double> &dL_dz) = 0;

protected:
    void add_parameter_group(const std::string& name, int size);
};

#endif