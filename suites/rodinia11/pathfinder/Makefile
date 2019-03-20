include ../env.mk
TARGET = pathfinder
NVCC_FLAGS = -g -G -rdc=true -Xptxas -v -I$(CUDA_BASE_DIR)/include/ $(EXTRA_NVCC_FLAGS)
LINK_FLAGS = -std=c++11 -g -G $(GENCODE) -lcudart -lcudadevrt -lm $(EXTRA_LINK_FLAGS)
#
#CC := $(CUDA_DIR)/bin/nvcc

#INCLUDE := $(CUDA_DIR)/include

SRC = pathfinder.cu

all: clean $(TARGET) install

$(TARGET):
	$(NVCC) $(KERNEL_DIM) $(SRC) -o $(TARGET) $(NVCC_FLAGS) -L$(CUDA_LIB_DIR) $(LINK_FLAGS)

install:
	mkdir -p $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/
	cp $(TARGET) $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/

test:
	./pathfinder 100000 100 20 

golden:
	./pathfinder 100000 100 20 >golden_stdout.txt 2>golden_stderr.txt

clean:
	rm -f *.o *~ $(TARGET)

clobber: clean
	rm -f sassifi-inst-counts.txt sassifi-scalar-insts.txt golden* output.out

occupancy:$(TARGET)
	$(CUDA_BIN_DIR)/nvprof --devices 0 --metrics achieved_occupancy --concurrent-kernels off ./$(TARGET) 100000 100 20
