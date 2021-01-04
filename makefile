CUDA_ROOT_DIR=/usr/local/cuda

CC=gcc
NVCC=nvcc

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

.PHONY: clean

# Link .o files to get target executable
main: $(OBJS)
	$(NVCC) -arch=sm_60 $^ -o $@ $(CUDA_LIB_DIR) $(CUDA_INC_DIR) $(CUDA_LINK_LIBS)

# Compile main.cu into main.o
main.o: main.cu
	$(NVCC) -arch=sm_60 -dc $^ -o $@

# Compile .cpp and .cu files into .o files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu 
	$(NVCC) -arch=sm_60 -dc $^ -o $@

clean: 
	@rm main $(OBJS)
