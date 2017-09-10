#include ../../common/make.config
TARGET = hotspot
SUITE_NAME = rodinia

OPTION = none
KERNEL_DIM=-DRD_WG_SIZE_0=24 

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
CFLAGS = -O3
CXXFLAGS = -O3
export PATH := $(CCDIR)/bin/:$(PATH)
export LD_LIBRARY_PATH := $(CCDIR)/lib64:$(LD_LIBRARY_PATH):$(CUDA_LIB_DIR):$(CUPTI_LIB_DIR)

AFTER_REG = -Xptxas --sassi-inst-after="reg-writes" 
AFTER_MEM = -Xptxas --sassi-inst-after="memory" 
AFTER_REG_MEM = -Xptxas --sassi-inst-after="reg-writes\,memory" 
BEFORE_ALL = -Xptxas --sassi-inst-before="all"
BEFORE_COND_BRANCHES = -Xptxas --sassi-inst-before="cond-branches"
BEFORE_MEM = -Xptxas --sassi-inst-before="memory"
BEFORE_REGS = -Xptxas --sassi-inst-before="reg-writes\,reg-reads"

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
EXTRA_NVCC_FLAGS = $(AFTER_REG_MEM) $(AFTER_REG_MEM_INFO) $(BRANCH_AROUND) 
EXTRA_LINK_FLAGS = -L$(INST_LIB_DIR) -linstinjector $(CUPTI) 
endif
 
ifeq (${OPTION},rf_injector)
EXTRA_NVCC_FLAGS = $(BEFORE_ALL) $(BEFORE_REG_MEM_INFO) 
EXTRA_LINK_FLAGS = -L$(INST_LIB_DIR) -lrfinjector $(CUPTI)
endif

ifeq (${OPTION},cfg)
EXTRA_NVCC_FLAGS = -Xptxas --sassi-function-entry -Xptxas --sassi-bb-entry
EXTRA_LINK_FLAGS = -L$(INST_LIB_DIR) -lcfg $(CUPTI)
endif

ifeq (${OPTION},fritz_cfg)
EXTRA_NVCC_FLAGS = -Xptxas --sassi-function-entry -Xptxas --sassi-bb-entry
EXTRA_LINK_FLAGS = -L$(INST_LIB_DIR) -lfritz_cfg $(CUPTI)
endif

ifeq (${OPTION},fritz_profiler)
EXTRA_NVCC_FLAGS = $(LDWRAP) -lineinfo -Xptxas --sassi-function-entry -Xptxas --sassi-bb-entry
EXTRA_LINK_FLAGS = $(LDWRAP) -L$(INST_LIB_DIR) -lfritz_profiler $(CUPTI) -L$(BOOST_HOME)/lib -lboost_regex -lcrypto -Xlinker -rpath,$(BOOST_HOME)/lib
endif

ifeq (${OPTION}, branch)
EXTRA_NVCC_FLAGS = $(LDWRAP) -lineinfo $(BEFORE_COND_BRANCHES) $(BEFORE_COND_BRANCH_INFO) 
EXTRA_LINK_FLAGS = $(LDWRAP) -L$(INST_LIB_DIR) -lbranch $(CUPTI) -L$(BOOST_HOME)/lib -lboost_regex -lcrypto -Xlinker -rpath,$(BOOST_HOME)/lib
endif

ifeq (${OPTION},memaccesses)
EXTRA_NVCC_FLAGS = $(BEFORE_ALL)  $(BEFORE_REG_MEM_INFO) -Xptxas --sassi-bb-entry#$(BEFORE_ALL) $(BEFORE_REG_MEM_INFO)
EXTRA_LINK_FLAGS = -L$(INST_LIB_DIR) -lmemaccesses $(CUPTI) 
endif

ifeq (${OPTION},memdiverge)
EXTRA_NVCC_FLAGS = $(BEFORE_MEM) $(BEFORE_MEM_INFO)
EXTRA_LINK_FLAGS = -L$(INST_LIB_DIR) -lmemdiverge $(CUPTI) 
endif

NVCC_FLAGS = -g -G -rdc=true -Xptxas -v -I$(CUDA_BASE_DIR)/include/ $(EXTRA_NVCC_FLAGS)
LINK_FLAGS = -std=c++11 -g -G $(GENCODE) -lcudart -lcudadevrt -lm $(EXTRA_LINK_FLAGS)
#
#CC := $(CUDA_DIR)/bin/nvcc

#INCLUDE := $(CUDA_DIR)/include

SRC = hotspot.cu

all: clean $(TARGET) install

$(TARGET):
	$(NVCC) $(KERNEL_DIM) $(SRC) -o $(TARGET) $(NVCC_FLAGS) -L$(CUDA_LIB_DIR) $(LINK_FLAGS)

install:
	mkdir -p $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/
	cp $(TARGET) $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/

test:
	./hotspot 64 4 4 ../data/hotspot/temp_64 ../data/hotspot/power_64 output.out

golden:
	./hotspot 64 4 4 ../data/hotspot/temp_64 ../data/hotspot/power_64 output.out >golden_stdout.txt 2>golden_stderr.txt

clean:
	rm -f *.o *~ $(TARGET)

clobber: clean
	rm -f sassifi-inst-counts.txt golden* output.out
