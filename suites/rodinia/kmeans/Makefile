#include ../../common/make.config

TARGET = kmeans
SUITE_NAME = rodinia
OPOPTION = none

GENCODE = -gencode arch=compute_35,code=sm_35
# SASSIFI_HOME, INST_LIB_DIR, CCDIR, and CUDA_BASE_DIR should be set 

# If you want to "intercept" main and exit, use this GCC-specific option.
# We only use this option for Fermi examples, where we don't have the luxury
# of using UVM.
LDWRAP := -Xlinker "--wrap=main" -Xlinker "--wrap=exit"


# SASSI/CUDA
CUDA_LIB_DIR = $(CUDA_BASE_DIR)/lib64
CUDA_BIN_DIR = $(CUDA_BASE_DIR)/bin
CUPTI_LIB_DIR = $(CUDA_BASE_DIR)/extras/CUPTI/lib64
CUPTI = -L$(CUPTI_LIB_DIR) -lcupti 
NVCC = $(CUDA_BIN_DIR)/nvcc

# The C/CPP compiler you want to use, and associated flags.
CC = gcc
CXX = g++
CFLAGS = -g -fopenmp -O2
CXXFLAGS = -O3
export PATH := $(CCDIR)/bin/:$(PATH)
export LD_LIBRARY_PATH := $(CCDIR)/lib64:$(LD_LIBRARY_PATH):$(CUDA_LIB_DIR):$(CUPTI_LIB_DIR)

AFTER_REG = -Xptxas --sassi-inst-after="reg-writes" 
AFTER_MEM = -Xptxas --sassi-inst-after="memory" 
AFTER_REG_MEM = -Xptxas --sassi-inst-after="reg-writes\,memory" 
BEFORE_ALL = -Xptxas --sassi-inst-before="all"
BEFORE_COND_BRANCHES = -Xptxas --sassi-inst-before="cond-branches"
BEFORE_MEM = -Xptxas --sassi-inst-before="memory"
BEFORE_REGS = -Xptxas --sassi-inst-before="reg-writes,reg-reads"

AFTER_REG_INFO = -Xptxas --sassi-after-args="reg-info"
AFTER_MEM_INFO = -Xptxas --sassi-after-args="mem-info"
AFTER_REG_MEM_INFO = -Xptxas --sassi-after-args="reg-info\,mem-info"
BEFORE_COND_BRANCH_INFO = -Xptxas --sassi-before-args="cond-branch-info"
BEFORE_MEM_INFO = -Xptxas --sassi-before-args="mem-info"
BEFORE_REG_INFO = -Xptxas --sassi-before-args="reg-info"
BEFORE_REG_MEM_INFO = -Xptxas --sassi-before-args="reg-info\,mem-info"

BRANCH_AROUND = -Xptxas --sassi-iff-true-predicate-handler-call

ifeq (${OPTION},profiler)
EXTRA_NVCC_FLAGS = $(BEFORE_ALL) $(BEFORE_REG_MEM_INFO)
EXTRA_LINK_FLAGS = -L$(INST_LIB_DIR) -lprofiler $(CUPTI) 
endif

ifeq (${OPTION},inst_injector)
EXTRA_NVCC_FLAGS = $(AFTER_REG_MEM) $(AFTER_REG_MEM_INFO) $(BRANCH_AROUND)  -Xptxas --sassi-function-entry -Xptxas --sassi-bb-entry
EXTRA_LINK_FLAGS = -L$(INST_LIB_DIR) -linstinjector $(CUPTI) 
endif
 
ifeq (${OPTION},rf_injector)
EXTRA_NVCC_FLAGS = $(BEFORE_ALL) $(BEFORE_REG_MEM_INFO) 
EXTRA_LINK_FLAGS = -L$(INST_LIB_DIR) -lrfinjector $(CUPTI)
endif

ifeq (${OPTION},fritz_cfg)
EXTRA_NVCC_FLAGS = -Xptxas --sassi-function-entry -Xptxas --sassi-bb-entry
EXTRA_LINK_FLAGS = -L$(INST_LIB_DIR) -lfritz_cfg $(CUPTI)
endif

ifeq (${OPTION},fritz_profiler)
EXTRA_NVCC_FLAGS = $(LDWRAP) -lineinfo -Xptxas --sassi-function-entry -Xptxas --sassi-bb-entry
EXTRA_LINK_FLAGS = $(LDWRAP) -L$(INST_LIB_DIR) -lfritz_profiler $(CUPTI) -L$(BOOST_HOME)/lib -lboost_regex -lcrypto -Xlinker -rpath,$(BOOST_HOME)/lib
endif

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
	./kmeans -o -i ../data/kmeans/100_34.txt

golden:
	./kmeans -o -i ../data/kmeans/100_34.txt >golden_stdout.txt 2>golden_stderr.txt

clean:
	rm -f *.o *~ kmeans kmeans_cuda.linkinfo

clobber: clean
	rm -f sassifi-inst-counts.txt golden*
