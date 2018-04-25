include ../env.mk
TARGET = reduction
#NVCC_FLAGS = -O3 -g -rdc=true -Xptxas -v
#LINK_FLAGS = -std=c++11 -g 
NVCC_FLAGS = -O3 -g -rdc=true -Xptxas -v -I$(CUDA_BASE_DIR)/include/ $(CUDA_SAMPLES_INC) $(EXTRA_NVCC_FLAGS)
LINK_FLAGS = -std=c++11 -m64 -g $(GENCODE) -lcudart -lcudadevrt -lm $(EXTRA_LINK_FLAGS)

all: clean $(TARGET) install
#all: clean $(TARGET) 

$(TARGET): 
	$(NVCC) -ccbin g++ -o reduction_kernel.o -c reduction_kernel.cu $(NVCC_FLAGS)  -L$(CUDA_LIB_DIR) $(LINK_FLAGS)
	$(NVCC) -ccbin g++ -m64 $(GENCODE) $(CUDA_SAMPLES_INC) reduction.o -c reduction.cpp
	$(NVCC) $(NVCC_FLAGS) -o reduction reduction.o reduction_kernel.o $(LINK_FLAGS)

install:
	mkdir -p $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/
	cp $(TARGET) $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/

test:
	./reduction 

golden:
	./reduction >golden_stdout.txt 2>golden_stderr.txt

clean:
	rm -f *.o *~ reduction

clobber: clean
	rm -f sassifi-inst-counts.txt sassifi-scalar-insts.txt golden* *_regcount.p stdout* stderr*
