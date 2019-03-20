include ../env.mk
TARGET = hybridsort


NVCC_FLAGS = -g -G -rdc=true -Xptxas -v -I$(CUDA_BASE_DIR)/include/ $(EXTRA_NVCC_FLAGS)
LINK_FLAGS = -std=c++11 $(GENCODE) -lcudart -lcudadevrt $(EXTRA_LINK_FLAGS)
CC := $(CUDA_DIR)/bin/nvcc

override VERIFY = -DVERIFY

ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

ifdef TIMER
override TIMER = -DTIMER
endif
all: clean $(TARGET) install

$(TARGET): main.cu  bucketsort.cu bucketsort.cuh bucketsort_kernel.cu histogram1024_kernel.cu  mergesort.cu mergesort.cuh mergesort_kernel.cu
	$(NVCC) $(VERIFY) $(OUTPUT) $(TIMER) $(HISTO_WG_SIZE_0) $(BUCKET_WG_SIZE_0) $(BUCKET_WG_SIZE_1) \
	$(MERGE_WG_SIZE_0) $(MERGE_WG_SIZE_1) bucketsort.cu mergesort.cu main.cu -o $(TARGET) \
	$(NVCC_FLAGS) $(LINK_FLAGS)

install:
	mkdir -p $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/
	cp $(TARGET) $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/

test:
	./$(TARGET) ../data/hybridsort/500000.txt

golden:
	./$(TARGET) ../data/hybridsort/500000.txt > golden_stdout.txt 2>golden_stderr.txt

clean:
	rm -f *.o hybridsort

clobber: clean
	rm -f sassifi-inst-counts.txt golden*
