include ../env.mk
TARGET = gaussian
DATASET_DIR=../data/gaussian
NVCC_FLAGS = -g -rdc=true -Xptxas -v -I$(CUDA_BASE_DIR)/include/ $(EXTRA_NVCC_FLAGS)
LINK_FLAGS = -std=c++11 -g $(GENCODE) -L$(CUDA_LIB_DIR) -lcudart -lcudadevrt -lm $(EXTRA_LINK_FLAGS)

SRC = gaussian.cu

all: clean $(TARGET) install


$(TARGET): $(SRC)
	$(NVCC) $(KERNEL_DIM) $(SRC) -o $(TARGET) $(NVCC_FLAGS) $(LINK_FLAGS)

install:
	mkdir -p $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/
	cp $(TARGET) $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/

test:
	./gaussian -f ../data/gaussian/matrix1024.txt

golden:
	./gaussian -f $(DATASET_DIR)/matrix1024.txt >golden_stdout.txt 2>golden_stderr.txt
#	./gaussian -s 16 >>golden_stdout.txt 2>>golden_stderr.txt

release: $(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) -I$(INCLUDE)  

clean:
	rm -f gaussian

clobber: clean
	rm -f sassifi-inst-counts.txt golden* 
