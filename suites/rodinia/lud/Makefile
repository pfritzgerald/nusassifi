include ../env.mk
TARGET = lud_cuda

NVCC_FLAGS=-rdc=true -Xptxas -v $(GENCODE) $(EXTRA_NVCC_FLAGS)
LINK_FLAGS=-std=c++11 -g $(GENCODE) -L$(CUDA_LIB_DIR) -lcudart -lcudadevrt $(EXTRA_LINK_FLAGS)

all: clean $(TARGET) install 

$(TARGET):
	cd cuda; make NVCC_FLAGS='$(NVCC_FLAGS)' LINK_FLAGS='$(LINK_FLAGS)'

install:
	mkdir -p $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/
	cp cuda/$(TARGET) $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/

test:
	cuda/lud_cuda -s 512 -v
golden:
	cuda/lud_cuda -s 512 -v > golden_stdout.txt 2>golden_stderr.txt
clean:
	cd cuda; make clean

clobber: clean
	rm -f sassifi-inst-counts.txt golden*
