# CudaNeuralNet

This is a simple neural network implementation using CUDA.

## Building the Project

This project uses a `makefile` to manage the build process. You can compile the project for different NVIDIA GPU architectures.

### Prerequisites

*   NVIDIA CUDA Toolkit (nvcc)
*   A compatible NVIDIA GPU

### Compilation Targets

You can specify the target GPU architecture when running `make`.

*   **Default (Fat Binary for V100 and GTX 1080):**
    This is the recommended option for portability. It creates a single executable that can run on both Tesla V100 and GeForce GTX 1080 GPUs.

    ```bash
    make
    ```

*   **Tesla V100 (Volta):**
    To compile specifically for a Tesla V100 GPU (`sm_70`):

    ```bash
    make TARGET_GPU=V100
    ```

*   **GeForce GTX 1080 (Pascal):**
    To compile specifically for a GeForce GTX 1080 GPU (`sm_61`):

    ```bash
    make TARGET_GPU=GTX1080
    ```

### Cleaning the Build

To remove the compiled object files and the final executable:

```bash
make clean
```

