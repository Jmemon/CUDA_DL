

In this file, we work out how to have a general module type (needed for adam). The hard part being a structure for the general module parameters. It could be a hashmap of strings to tensor pointers. Then the create functions for each module take on a lot more significance, as these have to ensure the right parameters are created in teh hashamp. Then forward/backward functions have to handle the hashmap.
Maybe include `param_names: string[]` and `param_values: Map<string, DoubleTensor>` in the module struct. init/forward/backward/free functions have to ensure everything is set up correctly. param_names can be hardcoded in the struct definitions(? is this possible?) for each module since these are the same, param_values would map the names to initialized tensors (inited based on arch hyperparams). 
Also include analogous maps for hyperparams: `hyperparam_names: string[]` and `hyperparam_values: Map<string, double>`.
Also include a `name: string` field in the module struct.
Module has validate param name function. This could be called by get/set/free functions. These each defined as part of Module header as well, taking (param_value: DoubleTensor, param_name: string, new_param_value: DoubleTensor) (as needed) as arguments. Check the param_name is valid and then if so check existing tensor dims and types to make sure new tensor is compatible.
Similarly include map for param_grad_buffers, input_buffers, input_grad_buffers. These are used for backprop and weight updates.

This also means each module type does not require a separate struct, as the module struct can be used for all module types. They only need to define the create/forward/backward/free functions for themselves. A validation function to ensure module struct is a linear/conv2d/etc. also should be created (check it contains the right parameters, hyperparams, and name).





And we also work out how to give adam the ability to store moment estimates for gradient updates, and how to update the weights.