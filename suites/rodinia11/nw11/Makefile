include ../env.mk
TARGET = nw
CFLAGS = -g -O3
CXXFLAGS = -g -O3
export PATH := $(CCDIR)/bin/:$(PATH)
export LD_LIBRARY_PATH := $(CCDIR)/lib64:$(LD_LIBRARY_PATH):$(CUDA_LIB_DIR):$(CUPTI_LIB_DIR)

NVCC_FLAGS = -g -rdc=true -Xptxas -v -I$(CUDA_BASE_DIR)/include/ $(EXTRA_NVCC_FLAGS)
LINK_FLAGS = -std=c++11 -g $(GENCODE) -lcudart -lcudadevrt -lm $(EXTRA_LINK_FLAGS)
#

#CC := $(CUDA_DIR)/bin/nvcc

#INCLUDE := $(CUDA_DIR)/include

SRC = needle.cu

#EXE = needle

all: clean $(TARGET) install

$(TARGET):
	$(NVCC) $(SRC) -o $(TARGET) $(NVCC_FLAGS) -L$(CUDA_LIB_DIR) $(LINK_FLAGS)

install:
	mkdir -p $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/
	cp $(TARGET) $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/

test:
	./nw 8192 10

golden:
	./nw 8192 10 >golden_stdout.txt 2>golden_stderr.txt

clean: $(SRC)
	rm -f $(TARGET) *.o *~

clobber: clean
	rm -f sassifi-inst-counts.txt golden* result.txt
