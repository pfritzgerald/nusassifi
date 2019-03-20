include ../env.mk

OUTPUT=Y
TARGET = lavaMD
CFLAGS = -g -fopenmp -O2
CXXFLAGS = -O3
export PATH := $(CCDIR)/bin/:$(PATH)
export LD_LIBRARY_PATH := $(CCDIR)/lib64:$(LD_LIBRARY_PATH):$(CUDA_LIB_DIR):$(CUPTI_LIB_DIR)

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
	./lavaMD -boxes1d 5

golden:
	./lavaMD -boxes1d 5 >golden_stdout.txt 2>golden_stderr.txt

# delete all object and executable files
clean:
	rm -f *.o ./kernel/*.o lavaMD \
	  ./util/num/*.o ./util/timer/*.o ./util/device/*.o

clobber: clean
	rm -f sassifi-inst-counts.txt golden* result.txt
