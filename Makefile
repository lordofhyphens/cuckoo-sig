# Change this to point to wherever CUDA is installed on your local system.
CUDA_PATH := /opt/net/apps/cuda
NVCC=${CUDA_PATH}/bin/nvcc
CXX=g++-4.8

CC_FLAGS := -I./hemi -I${CUDA_PATH}/include

NVCC_FLAGS := ${CC_FLAGS} -arch=sm_21  -ccbin ${CXX}
ARCH := $(shell getconf LONG_BIT)

LIB_FLAGS_32 := -L$(CUDA_PATH)/lib
LIB_FLAGS_64 := -L$(CUDA_PATH)/lib64

LIB_FLAGS := $(LIB_FLAGS_$(ARCH)) -lcudart

.PHONY: sigtable clean
all: sigtable

clean: 
	rm -f sigtable_host_g++ sigtable_host_nvcc sigtable_device 
sigtable: sigtable_host_nvcc sigtable_device 
	@for prog in $^; do \
		echo "---------\nRunning $$prog\n---------"; \
		./$$prog; \
	done

sigtable_device: sigtable_test.cpp sigtable.h
	@${NVCC} ${LIB_FLAGS} -x cu ${CC_FLAGS} ${NVCC_FLAGS} $< -o $@

sigtable_host_g++: sigtable_test.cpp sigtable.h
	@${CXX} ${LIB_FLAGS} ${CC_FLAGS} $< -o $@

sigtable_host_nvcc: sigtable_test.cpp sigtable.h
	@${NVCC} ${LIB_FLAGS} -x c++ ${CC_FLAGS} ${NVCC_FLAGS} $< -o $@
