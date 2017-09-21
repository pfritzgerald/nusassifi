include ../env.mk
TARGET = hotspot
NVCC_FLAGS = -g -G -rdc=true -Xptxas -v -I$(CUDA_BASE_DIR)/include/ $(EXTRA_NVCC_FLAGS)
LINK_FLAGS = -std=c++11 -g -G $(GENCODE) -lcudart -lcudadevrt -lm $(EXTRA_LINK_FLAGS)
#
#CC := $(CUDA_DIR)/bin/nvcc

#INCLUDE := $(CUDA_DIR)/include

SRC = hotspot.cu

all: clean $(TARGET) install

$(TARGET):
	$(NVCC) $(KERNEL_DIM) $(SRC) -o $(TARGET) $(NVCC_FLAGS) -L$(CUDA_LIB_DIR) $(LINK_FLAGS)

install:
	mkdir -p $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/
	cp $(TARGET) $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/

test:
	./hotspot 64 4 4 ../data/hotspot/temp_64 ../data/hotspot/power_64 output.out

golden:
	./hotspot 64 4 4 ../data/hotspot/temp_64 ../data/hotspot/power_64 output.out >golden_stdout.txt 2>golden_stderr.txt

clean:
	rm -f *.o *~ $(TARGET)

clobber: clean
	rm -f sassifi-inst-counts.txt golden* output.out
