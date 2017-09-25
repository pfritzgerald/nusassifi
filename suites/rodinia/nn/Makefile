include ../env.mk
TARGET = nn

NVCC_FLAGS = -rdc=true -Xptxas -v $(GENCODE) $(EXTRA_NVCC_FLAGS)
LINK_FLAGS = -std=c++11 -g $(GENCODE) -L$(CUDA_LIB_DIR) $(EXTRA_LINK_FLAGS) -lcudart -lcudadevrt 

all : clean $(TARGET) install

$(TARGET) : nn_cuda.cu
	$(NVCC) nn_cuda.cu $(NVCC_FLAGS) -o nn  $(LINK_FLAGS)

install:
	mkdir -p $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/
	cp $(TARGET) $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/

test:
	./nn filelist_4 -r 5 -lat 30 -lng 90

golden:
	./nn filelist_4 -r 5 -lat 30 -lng 90 >golden_stdout.txt 2>golden_stderr.txt


hurricane_gen : hurricane_gen.c
	$(CC) -g -O3 -Wall -o $@ $< -lm

clean :
	rm -rf *.o nn hurricane_gen

clobber: clean
	rm -f sassifi-inst-counts.txt golden*

#data :
#	mkdir data
#	./gen_dataset.sh
