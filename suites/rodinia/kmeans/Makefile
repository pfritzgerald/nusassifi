include ../env.mk

TARGET = kmeans
CFLAGS = -g -fopenmp -O2
CXXFLAGS = -O3
export PATH := $(CCDIR)/bin/:$(PATH)
export LD_LIBRARY_PATH := $(CCDIR)/lib64:$(LD_LIBRARY_PATH):$(CUDA_LIB_DIR):$(CUPTI_LIB_DIR)

NVCC_FLAGS = -g  -rdc=true -Xptxas -v -I$(CUDA_BASE_DIR)/include/ $(EXTRA_NVCC_FLAGS) $(GENCODE)
LINK_FLAGS = -std=c++11 -g $(GENCODE) -lcudart -lcudadevrt -lm $(EXTRA_LINK_FLAGS)
#
# C compiler
# #CC = gcc
#CC_FLAGS = -g -fopenmp -O2 

# CUDA compiler
#NVCC = $(CUDA_DIR)/bin/nvcc
#NVCC_FLAGS = -I$(CUDA_DIR)/include 

# 'make dbg=1' enables NVCC debugging
#ifeq ($(dbg),1)
#	NVCC_FLAGS += -g -O0
#else
#	NVCC_FLAGS += -O2
#endif

# 'make emu=1' compiles the CUDA kernels for emulation
#ifeq ($(emu),1)
#	NVCC_FLAGS += -deviceemu
#endif

all: clean $(TARGET) install

$(TARGET): cluster.o getopt.o kmeans.o kmeans_clustering.o kmeans_cuda.o rmse.o
	$(NVCC) -Xlinker "$(CFLAGS)" cluster.o getopt.o kmeans.o kmeans_clustering.o kmeans_cuda.o rmse.o -o kmeans -L$(CUDA_LIB_DIR) $(LINK_FLAGS)

%.o: %.[ch]
	$(CC) $(CFLAGS) $< -c

kmeans_cuda.o: kmeans_cuda.cu 
	$(NVCC) $(NVCC_FLAGS) -c kmeans_cuda.cu

install:
	mkdir -p $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/
	cp $(TARGET) $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/

test:
	./kmeans -o -i ../data/kmeans/100000_34.txt

golden:
	./kmeans -o -i ../data/kmeans/100000_34.txt >golden_stdout.txt 2>golden_stderr.txt

clean:
	rm -f *.o *~ kmeans kmeans_cuda.linkinfo

clobber: clean
	rm -f sassifi-inst-counts.txt golden*
