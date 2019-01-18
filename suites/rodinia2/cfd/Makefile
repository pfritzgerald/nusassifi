include ../env.mk
TARGET = euler3d
#GENCODE = -gencode arch=compute_20,code=sm_20

# Determine the correct version of the cutil library
CUTIL_LIB = # -lcutil
ifeq ($(shell uname -m), x86_64)
     ifeq ($(shell if test -e $(SDK_DIR)/lib/libcutil_x86_64.a; then echo T; else echo F; fi), T)
        CUTIL_LIB = #-lcutil_x86_64
     endif
endif

NVCC_FLAGS = -g -G -rdc=true -Xptxas -v -I$(CUDA_BASE_DIR)/include $(EXTRA_NVCC_FLAGS)
LINK_FLAGS = -std=c++11 -g -G $(GENCODE) -lcudart -lcudadevrt $(EXTRA_LINK_FLAGS)

#all: euler3d euler3d_double  pre_euler3d  pre_euler3d_double 
all: clean $(TARGET) install

$(TARGET): euler3d.cu
	$(NVCC) $(KERNEL_DIM) -O2 euler3d.cu -o euler3d \
	$(NVCC_FLAGS) $(CUDA_SAMPLES_INC)  \
	$(LINK_FLAGS) -L$(CUDA_BASE_DIR)/samples/common/lib $(CUTIL_LIB) 

install:
	mkdir -p $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/
	cp $(TARGET) $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/

test:
	./$(TARGET) ../data/cfd/fvcorr.domn.097K

golden:
	./$(TARGET) ../data/cfd/fvcorr.domn.097K >golden_stdout.txt 2>golden_stderr.txt

clean:
	rm -f euler3d euler3d_double pre_euler3d pre_euler3d_double *.linkinfo

clobber: clean
	rm -f sassifi-inst-counts.txt golden* result.txt density density_gravity momentum
euler3d_double: euler3d_double.cu
	nvcc -Xptxas -v -O3 --gpu-architecture=compute_20 --gpu-code=compute_20 euler3d_double.cu -o euler3d_double -I$(CUDA_SDK_PATH)/common/inc  -L$(CUDA_SDK_PATH)/lib $(CUTIL_LIB)


pre_euler3d: pre_euler3d.cu
	nvcc -Xptxas -v -O3 --gpu-architecture=compute_20 --gpu-code=compute_20 pre_euler3d.cu -o pre_euler3d -I$(CUDA_SAMPLES_INC)  -L$(CUDA_BASE_DIR)/samples/common/lib $(CUTIL_LIB)

pre_euler3d_double: pre_euler3d_double.cu
	nvcc -Xptxas -v -O3 --gpu-architecture=compute_20 --gpu-code=compute_20 pre_euler3d_double.cu -o pre_euler3d_double -I$(CUDA_SDK_PATH)/common/inc  -L$(CUDA_SDK_PATH)/lib $(CUTIL_LIB)
