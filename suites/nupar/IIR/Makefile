TARGET = parIIR
SUITE_NAME = nupar

OPTION = none

GENCODE = -gencode arch=compute_35,code=sm_35 #-gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 

# SASSIFI_HOME, INST_LIB_DIR, CCDIR, and CUDA_BASE_DIR should be set 

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
BEFORE_REG_MEM = -Xptxas --sassi-inst-before="reg-writes\,memory"

AFTER_REG_INFO = -Xptxas --sassi-after-args="reg-info"
AFTER_MEM_INFO = -Xptxas --sassi-after-args="mem-info"
AFTER_REG_MEM_INFO = -Xptxas --sassi-after-args="reg-info\,mem-info"
BEFORE_COND_BRANCH_INFO = -Xptxas --sassi-before-args="cond-branch-info"
BEFORE_REG_MEM_INFO = -Xptxas --sassi-before-args="reg-info\,mem-info"

BRANCH_AROUND = -Xptxas --sassi-iff-true-predicate-handler-call

ifeq (${OPTION}, profiler)
EXTRA_NVCC_FLAGS = $(BEFORE_ALL) $(BEFORE_REG_MEM_INFO)
EXTRA_LINK_FLAGS = -L$(INST_LIB_DIR) -lprofiler $(CUPTI)
endif

ifeq (${OPTION}, inst_injector)
EXTRA_NVCC_FLAGS = $(AFTER_REG_MEM) $(AFTER_REG_MEM_INFO) $(BRANCH_AROUND) 
EXTRA_LINK_FLAGS = -L$(INST_LIB_DIR) -linstinjector $(CUPTI)
endif
 
ifeq (${OPTION}, rf_injector)
EXTRA_NVCC_FLAGS = $(BEFORE_ALL) $(BEFORE_REG_MEM_INFO) 
EXTRA_LINK_FLAGS = -L$(INST_LIB_DIR) -lrfinjector $(CUPTI)
endif

ifeq (${OPTION},bbv_profiler)
EXTRA_NVCC_FLAGS = $(AFTER_REG_MEM) $(AFTER_REG_MEM_INFO) -Xptxas --sassi-bb-entry $(BRANCH_AROUND) $(GENCODE)
EXTRA_LINK_FLAGS = -L$(INST_LIB_DIR) -lbbv_profiler $(CUPTI) 
endif

ifeq (${OPTION},mem_profiler)
EXTRA_NVCC_FLAGS = $(AFTER_REG_MEM)  $(AFTER_REG_MEM_INFO)  $(BRANCH_AROUND) $(GENCODE)
EXTRA_LINK_FLAGS = -L$(INST_LIB_DIR) -lmem_profiler $(CUPTI) 
endif

#NVCC_FLAGS = -O3 -g -rdc=true -Xptxas -v
#LINK_FLAGS = -std=c++11 -g 
VERBOSE = --ptxas-options=-v
NVCC_FLAGS = $(VERBOSE) -O3 -g -rdc=true -Xptxas -v -I$(CUDA_BASE_DIR)/include/ $(EXTRA_NVCC_FLAGS)
#NVCC_FLAGS = -O3 -g -rdc=true -Xptxas -v -I$(CUDA_BASE_DIR)/include/ $(EXTRA_NVCC_FLAGS)
LINK_FLAGS = -std=c++11 -m64 -g $(GENCODE) -lcudart -lcudadevrt -lm $(EXTRA_LINK_FLAGS)

all: clean $(TARGET) install
#all: clean $(TARGET) 

$(TARGET):parIIR.cu
	$(NVCC) -ccbin g++ -o $@ $^ $(NVCC_FLAGS) -I/shared/apps/cuda7.0/samples/common/inc/ -L$(CUDA_LIB_DIR) $(LINK_FLAGS)

install:
	mkdir -p $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/
	cp $(TARGET) $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/

test:
	./parIIR 8192

golden:
	./parIIR 8192 >golden_stdout.txt 2>golden_stderr.txt

clean:
	rm -f *.o *~ parIIR

clobber: clean
	rm -f sassifi-inst-counts.txt sassifi-scalar-insts.txt golden* *_regcount.p stdout* stderr*

occupancy:
	$(CUDA_BIN_DIR)/nvprof --devices 0 --metrics achieved_occupancy --concurrent-kernels off ./parIIR 8192

