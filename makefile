CUDA_ROOT_DIR=/usr/local/cuda

CC=gcc
NVCC=nvcc

# --- GPU Architecture Configuration ---
# Set the target GPU architecture. This can be overridden from the command line.
# Examples:
#   make TARGET_GPU=V100     # Compile specifically for Tesla V100 (sm_70)
#   make TARGET_GPU=GTX1080  # Compile specifically for GeForce GTX 1080 (sm_61)
#   make                   # By default, compiles a 'fat binary' for both
#
# A 'fat binary' includes code for multiple GPU architectures. This increases
# portability and is generally recommended unless you need to minimize binary
# size or compilation time for a single, known target architecture.
#
TARGET_GPU ?= ALL

# NVIDIA GPU compute capabilities:
#   - Tesla V100:      Volta architecture,  sm_70
#   - GeForce GTX 1080:  Pascal architecture, sm_61
V100_ARCH    := -gencode arch=compute_70,code=sm_70
GTX1080_ARCH := -gencode arch=compute_61,code=sm_61

NVCC_GENCODE_FLAGS :=
ifeq ($(TARGET_GPU), V100)
	NVCC_GENCODE_FLAGS := $(V100_ARCH)
else ifeq ($(TARGET_GPU), GTX1080)
	NVCC_GENCODE_FLAGS := $(GTX1080_ARCH)
else ifeq ($(TARGET_GPU), ALL)
	NVCC_GENCODE_FLAGS := $(V100_ARCH) $(GTX1080_ARCH)
else
    $(error "Invalid TARGET_GPU specified: '$(TARGET_GPU)'. Supported values are V100, GTX1080, ALL.")
endif
# --- End GPU Architecture Configuration ---

# CUDA library directory:
CUDA_LIB_DIR= -L $(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I $(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart

SRC_DIR=src
INC_DIR=include
OBJ_DIR=bin

SRCS=main.cu $(SRC_DIR)/NeuralNet.cu $(SRC_DIR)/Activation.cu $(SRC_DIR)/Matrix.cu $(SRC_DIR)/Loss.cu
DEPS=$(INC_DIR)/NeuralNet.h $(INC_DIR)/Activation.cuh $(INC_DIR)/Matrix.cuh $(INC_DIR)/Loss.cuh
OBJS=$(OBJ_DIR)/main.o $(OBJ_DIR)/NeuralNet.o $(OBJ_DIR)/Activation.o $(OBJ_DIR)/Matrix.o $(OBJ_DIR)/Loss.o 

.PHONY: clean all

all: main

# Link .o files to get target executable
main: $(OBJS)
	@echo "--- Linking for target GPU(s): $(TARGET_GPU) ---"
	$(NVCC) $(NVCC_GENCODE_FLAGS) $^ -o $@ $(CUDA_LIB_DIR) $(CUDA_INC_DIR) $(CUDA_LINK_LIBS)

# Compile main.cu into main.o
$(OBJ_DIR)/main.o: main.cu
	$(NVCC) $(NVCC_GENCODE_FLAGS) -dc $^ -o $@

# Compile .cpp and .cu files into .o files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu 
	$(NVCC) $(NVCC_GENCODE_FLAGS) -dc $^ -o $@

clean: 
	@echo "--- Cleaning build objects and executable ---"
	@rm main $(OBJS)
