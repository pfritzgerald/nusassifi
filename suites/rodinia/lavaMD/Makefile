#include ../../common/make.config

OUTPUT=Y
TARGET = lavaMD
SUITE_NAME = rodinia
OPOPTION = none

GENCODE = -gencode arch=compute_35,code=sm_35
# SASSIFI_HOME, INST_LIB_DIR, CCDIR, and CUDA_BASE_DIR should be set 
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

ifeq (${OPTION},fritz_cfg)
EXTRA_NVCC_FLAGS = -Xptxas --sassi-function-entry -Xptxas --sassi-bb-entry
EXTRA_LINK_FLAGS = -L$(INST_LIB_DIR) -lfritz_cfg $(CUPTI)
endif

ifeq (${OPTION},fritz_profiler)
EXTRA_NVCC_FLAGS = $(LDWRAP) -lineinfo -Xptxas --sassi-function-entry -Xptxas --sassi-bb-entry
EXTRA_LINK_FLAGS = $(LDWRAP) -L$(INST_LIB_DIR) -lfritz_profiler $(CUPTI) -L$(BOOST_HOME)/lib -lboost_regex -lcrypto -Xlinker -rpath,$(BOOST_HOME)/lib
endif

ifeq (${OPTION},memaccesses)
EXTRA_NVCC_FLAGS = $(BEFORE_ALL)  $(BEFORE_REG_MEM_INFO) -Xptxas --sassi-bb-entry#$(BEFORE_ALL) $(BEFORE_REG_MEM_INFO)
EXTRA_LINK_FLAGS = -L$(INST_LIB_DIR) -lmemaccesses $(CUPTI) 
endif

NVCC_FLAGS = -rdc=true -Xptxas -v -I$(CUDA_BASE_DIR)/include/ $(EXTRA_NVCC_FLAGS) $(GENCODE)
LINK_FLAGS = -std=c++11 $(GENCODE) -lcudart -lcudadevrt -lm $(EXTRA_LINK_FLAGS)
#

# Example
# target: dependencies
	# command 1
	# command 2
          # .
          # .
          # .
	# command n

all: clean $(TARGET) install

ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

#C_C = gcc

OMP_LIB = -lgomp
OMP_FLAG = -fopenmp

#CUD_C = $(CUDA_DIR)/bin/nvcc
# OMP_FLAG = 	-Xcompiler paste_one_here
CUDA_FLAG = -arch sm_35

# link objects (binaries) together
$(TARGET):		main.o \
			./kernel/kernel_gpu_cuda_wrapper.o \
			./util/num/num.o \
			./util/timer/timer.o \
			./util/device/device.o
	$(NVCC)	$(KERNEL_DIM) main.o ./kernel/kernel_gpu_cuda_wrapper.o \
			./util/num/num.o ./util/timer/timer.o \
			./util/device/device.o \
			-lm -L$(CUDA_LIB_DIR) $(LINK_FLAGS) $(OMP_LIB) \
			-o lavaMD

# compile function files into objects (binaries)
main.o:		main.h main.c \
  		./kernel/kernel_gpu_cuda_wrapper.h \
		./kernel/kernel_gpu_cuda_wrapper.cu \
		./util/num/num.h ./util/num/num.c \
		./util/timer/timer.h ./util/timer/timer.c \
		./util/device/device.h ./util/device/device.cu
	$(CC)	$(KERNEL_DIM) $(OUTPUT) main.c -c -o main.o -O3

./kernel/kernel_gpu_cuda_wrapper.o:	./kernel/kernel_gpu_cuda_wrapper.h \
					./kernel/kernel_gpu_cuda_wrapper.cu
	$(NVCC) $(KERNEL_DIM) $(NVCC_FLAGS) ./kernel/kernel_gpu_cuda_wrapper.cu \
		-c -o ./kernel/kernel_gpu_cuda_wrapper.o  $(CUDA_FLAG)

./util/num/num.o:	./util/num/num.h ./util/num/num.c
	$(CC) ./util/num/num.c -c -o ./util/num/num.o -O3

./util/timer/timer.o:	./util/timer/timer.h ./util/timer/timer.c
	$(CC) ./util/timer/timer.c -c -o ./util/timer/timer.o -O3

./util/device/device.o:	./util/device/device.h ./util/device/device.cu
	$(NVCC)	$(NVCC_FLAGS) ./util/device/device.cu -c -o ./util/device/device.o

install:
	mkdir -p $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/
	cp $(TARGET) $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/

test:
	./lavaMD -boxes1d 2

golden:
	./lavaMD -boxes1d 2 >golden_stdout.txt 2>golden_stderr.txt

# delete all object and executable files
clean:
	rm -f *.o ./kernel/*.o lavaMD \
	  ./util/num/*.o ./util/timer/*.o ./util/device/*.o

clobber: clean
	rm -f sassifi-inst-counts.txt golden* result.txt
