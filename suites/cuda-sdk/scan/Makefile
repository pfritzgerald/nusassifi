include ../env.mk
TARGET = scan
#NVCC_FLAGS = -O3 -g -rdc=true -Xptxas -v
#LINK_FLAGS = -std=c++11 -g 
NVCC_FLAGS = -O3 -g -rdc=true -Xptxas -v -I$(CUDA_BASE_DIR)/include/ $(CUDA_SAMPLES_INC) $(EXTRA_NVCC_FLAGS)
LINK_FLAGS = -std=c++11 -m64 -g $(GENCODE) -lcudart -lcudadevrt -lm $(EXTRA_LINK_FLAGS)

all: clean $(TARGET) install
#all: clean $(TARGET) 

$(TARGET): 
	$(NVCC) -ccbin g++ -o scan.o -c scan.cu $(NVCC_FLAGS)  -L$(CUDA_LIB_DIR) $(LINK_FLAGS)
	$(NVCC) -ccbin g++ -m64 -po -o  main.o -c main.cpp $(CUDA_SAMPLES_INC)
	$(NVCC) -ccbin g++ -m64 -po -o scan_gold.o -c scan_gold.cpp $(CUDA_SAMPLES_INC)
	$(NVCC) $(NVCC_FLAGS) -o scan main.o scan.o scan_gold.o $(LINK_FLAGS)

install:
	mkdir -p $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/
	cp $(TARGET) $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/

test:
	./scan 

golden:
	./scan >golden_stdout.txt 2>golden_stderr.txt

clean:
	rm -f *.o *~ scan

clobber: clean
	rm -f sassifi-inst-counts.txt sassifi-scalar-insts.txt golden* *_regcount.p stdout* stderr*
