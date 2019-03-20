include ../env.mk
TARGET = srad_v2

NVCC_FLAGS = -rdc=true -Xptxas -v -I$(CUDA_BASE_DIR)/include/ $(EXTRA_NVCC_FLAGS) $(GENCODE)
LINK_FLAGS = -std=c++11 $(GENCODE) -lcudart -lcudadevrt $(EXTRA_LINK_FLAGS)
#
INCLUDE := $(CUDA_DIR)/include

SRC = srad.cu

all: clean $(TARGET) install

$(TARGET): $(SRC)
	$(NVCC) $(KERNEL_DIM) $(SRC) -o $(TARGET) $(NVCC_FLAGS) $(LINK_FLAGS) 

install:
	mkdir -p $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/
	cp $(TARGET) $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/

test:
	./srad_v2 2048 2048 0 127 0 127 0.5 2

golden:
	./srad_v2 2048 2048 0 127 0 127 0.5 2 >golden_stdout.txt 2>golden_stderr.txt

clean: $(SRC)
	rm -f $(TARGET) $(TARGET).linkinfo

clobber: clean
	rm -f sassifi-inst-counts.txt golden* result.txt
