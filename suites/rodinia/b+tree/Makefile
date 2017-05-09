#include ../../common/make.config
#KERNEL_DIM="-DRD_WG_SIZE_0=256"
TARGET = b+tree.out
SUITE_NAME = rodinia
OPTION = none
GENCODE = -gencode arch=compute_35,code=sm_35
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
EXTRA_NVCC_FLAGS = $(AFTER_REG_MEM) $(AFTER_REG_MEM_INFO) $(BRANCH_AROUND) 
EXTRA_LINK_FLAGS = -L$(INST_LIB_DIR) -linstinjector $(CUPTI) 
endif
 
ifeq (${OPTION},rf_injector)
EXTRA_NVCC_FLAGS = $(BEFORE_ALL) $(BEFORE_REG_MEM_INFO) 
EXTRA_LINK_FLAGS = -L$(INST_LIB_DIR) -lrfinjector $(CUPTI)
endif

NVCC_FLAGS = -g -rdc=true -Xptxas -v -I$(CUDA_BASE_DIR)/include/ $(EXTRA_NVCC_FLAGS) $(GENCODE)
LINK_FLAGS = -std=c++11 -g $(GENCODE) -L$(CUDA_LIB_DIR) -lcuda -lcudart -lcudadevrt -lm $(EXTRA_LINK_FLAGS)
#
CUDA_LIB =	-lcuda -lcudart
CUDA_FLAG = -arch sm_20

# ================================================================================100
#	EXECUTABLES (LINK OBJECTS TOGETHER INTO BINARY)
# ================================================================================100
all: clean $(TARGET) install
$(TARGET):	./main.o \
		./kernel/kernel_gpu_cuda_wrapper.o ./kernel/kernel_gpu_cuda_wrapper_2.o \
		./util/timer/timer.o ./util/num/num.o ./util/cuda/cuda.o
	$(NVCC)	./main.o \
		./kernel/kernel_gpu_cuda_wrapper.o ./kernel/kernel_gpu_cuda_wrapper_2.o \
		./util/timer/timer.o ./util/num/num.o ./util/cuda/cuda.o \
		-lm $(LINK_FKAGS) -o b+tree.out

install:
	mkdir -p $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/
	cp $(TARGET) $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/

test:
	./b+tree.out file ../data/b+tree/mil.txt command ../data/b+tree/command.txt

golden:
	./b+tree.out file ../data/b+tree/mil.txt command ../data/b+tree/command.txt \
	  >golden_stdout.txt 2>golden_stderr.txt

# ================================================================================100
#	OBJECTS (COMPILE SOURCE FILES INTO OBJECTS)
# ================================================================================100

# ================================================================================100
#	MAIN FUNCTION
# ================================================================================100

main.o:	./common.h ./main.h ./main.c
	$(CC)	$(KERNEL_DIM) ./main.c -c -o ./main.o -O3

# ================================================================================100
#	KERNELS
# ================================================================================100

./kernel/kernel_gpu_cuda_wrapper.o:	./common.h \
  					./kernel/kernel_gpu_cuda_wrapper.h \
					./kernel/kernel_gpu_cuda_wrapper.cu
	$(NVCC) $(KERNEL_DIM) ./kernel/kernel_gpu_cuda_wrapper.cu -c -o \
	  			./kernel/kernel_gpu_cuda_wrapper.o \
				-O3 $(NVCC_FLAGS)

./kernel/kernel_gpu_cuda_wrapper_2.o:	./common.h \
  					./kernel/kernel_gpu_cuda_wrapper_2.h \
					./kernel/kernel_gpu_cuda_wrapper_2.cu
	$(NVCC) $(KERNEL_DIM) ./kernel/kernel_gpu_cuda_wrapper_2.cu \
				-c -o ./kernel/kernel_gpu_cuda_wrapper_2.o \
				-O3 $(NVCC_FLAGS)

# ================================================================================100
#	UTILITIES
# ================================================================================100

./util/timer/timer.o:	./common.h \
  			./util/timer/timer.h ./util/timer/timer.c
	$(CC)	./util/timer/timer.c -c -o ./util/timer/timer.o -O3

./util/num/num.o:	./common.h ./util/num/num.h ./util/num/num.c
	$(CC)	./util/num/num.c -c -o ./util/num/num.o -O3

./util/cuda/cuda.o:	./common.h ./util/cuda/cuda.h ./util/cuda/cuda.cu
	$(NVCC)	./util/cuda/cuda.cu -c -o ./util/cuda/cuda.o -O3 $(NVCC_FLAGS)

# ================================================================================100
#	END
# ================================================================================100

# ================================================================================100
#	DELETE
# ================================================================================100

clean:
	rm -f *.o  ./kernel/*.o ./util/timer/*.o ./util/num/*.o ./util/cuda/*.o

clobber: clean
	rm -f sassifi-inst-counts.txt golden* output.txt

# ================================================================================100
#	END
# ================================================================================100
 
