include ../env.mk
TARGET = backprop

NVCC_FLAGS = -g -rdc=true -Xptxas -v -I$(CUDA_BASE_DIR)/include/ $(EXTRA_NVCC_FLAGS) $(GENCODE)
LINK_FLAGS = -std=c++11 -g $(GENCODE) -lcuda -lcudart -lcudadevrt -lm $(EXTRA_LINK_FLAGS)
#CC := $(CUDA_DIR)/bin/nvcc

all: clean $(TARGET) install

$(TARGET): backprop.o facetrain.o imagenet.o backprop_cuda.o 
	$(NVCC) $(CFLAGS) backprop.o facetrain.o imagenet.o backprop_cuda.o -o backprop $(LINK_FLAGS)

install:
	mkdir -p $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/
	cp $(TARGET) $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/

test:
	./backprop 16384

golden:
	./backprop 16384 >golden_stdout.txt 2>golden_stderr.txt


%.o: %.[ch]
	$(CC) $(CFLAGS) $< -c

facetrain.o: facetrain.c backprop.h
	$(CC) $(CFLAGS) facetrain.c -c
	
backprop.o: backprop.c backprop.h
	$(CC) $(CFLAGS) backprop.c -c

backprop_cuda.o: backprop_cuda.cu backprop.h
	$(NVCC) $(NVCC_FLAGS) -c backprop_cuda.cu

imagenet.o: imagenet.c backprop.h
	$(CC) $(CFLAGS) imagenet.c -c


clean:
	rm -f *.o *~ backprop backprop_cuda.linkinfo

clobber: clean
	rm -f sassifi-inst-counts.txt golden*
