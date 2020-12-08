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

SRCS=$(SRC_DIR)/main.cu $(SRC_DIR)/NeuralNet.cu $(SRC_DIR)/ParallelNN.cu
DEPS=$(INC_DIR)/NeuralNet.h $(INC_DIR)/ParallelNN.h
OBJS=$(OBJ_DIR)/main.o $(OBJ_DIR)/NeuralNet.o $(OBJ_DIR)/ParallelNN.o

.PHONY: clean

# Compile .cpp files into .o files
# $(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(INC_DIR)/%
# 	@$(CC) -c $< -o $@

# Compile .cpp and .cu files into .o files
$(OBJS): $(SRCS) $(DEPS)
	@$(NVCC) --gpu-architecture=sm_30 --device-c $< -o $@
	@$(NVCC) --gpu-architecture=sm_30 --device-link $@ -o $(OBJ_DIR)/link.o

# Link .o files to get target executable
main: $(OBJS)
	@$(CC) $< -o $@ $(CUDA_LIB_DIR) $(CUDA_INC_DIR) $(CUDA_LINK_LIBS)

clean: 
	@rm main $(OBJS)
