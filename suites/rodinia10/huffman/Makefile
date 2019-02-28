include ../env.mk
TARGET = pavle

#CUDA_INCLUDEPATH=/usr/local/cuda-6.5/include

NVCC_FLAGS=-O3 -arch=sm_35 -Xcompiler -m64 -g -G -rdc=true -Xptxas -v $(GENCODE) $(EXTRA_NVCC_FLAGS)
LINK_FLAGS=-arch=sm_35 -std=c++11 -g $(GENCODE) -L$(CUDA_LIB_DIR) -lcudart -lcudadevrt $(EXTRA_LINK_FLAGS)
GCC_OPTS=-O3 -Wall -Wextra -m64

OBJ = main_test_cu.o stats_logger.o vlc_kernel_sm64huff.o scan.o pack_kernels.o cpuencode.o
SRC = main_test_cu.cu

ifdef TESTING 
override TESTING = -DTESTING
endif

ifdef CACHECWLUT
override TESTING = -DCACHECWLUT
endif

all: clean $(TARGET) install

$(TARGET): $(OBJ) 
	$(NVCC) $(TESTING) $(CACHECWLUT)  $(OBJ) $(LINK_FLAGS) -o $(TARGET) 

vlc_kernel_sm64huff.o: vlc_kernel_sm64huff.cu 
	$(NVCC) $(TESTING) -c vlc_kernel_sm64huff.cu $(NVCC_FLAGS)

scan.o: scan.cu 
	$(NVCC) -c scan.cu $(NVCC_FLAGS)

#cpuencode.o: cpuencode.cu
#	$(NVCC) -c $(NVCC_FLAGS) cpuencode.cu

pack_kernels.o: pack_kernels.cu 
	$(NVCC) -c pack_kernels.cu $(NVCC_FLAGS)

main_test_cu.o: main_test_cu.cu cutil.h
	$(NVCC) $(TESTING) -c main_test_cu.cu $(NVCC_FLAGS) -I $(CUDA_BASE_DIR)/include 

.o:.cpp
	$(CXX) ++ $(GCC_OPTS) -c $@ -o $<

install:
	mkdir -p $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/
	cp $(TARGET) $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/

test:
	./pavle  ../data/huffman/test1024_H2.206587175259.in

golden:
	./pavle  ../data/huffman/test1024_H2.206587175259.in > golden_stdout.txt 2>golden_stderr.txt
	
clean:
	rm -f *.o $(TARGET)

clobber: clean
	rm -f sassifi-inst-counts.txt golden*
