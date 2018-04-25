include ../env.mk
TARGET = scalarProd
#NVCC_FLAGS = -O3 -g -rdc=true -Xptxas -v
#LINK_FLAGS = -std=c++11 -g 
NVCC_FLAGS = -O3 -g -rdc=true -Xptxas -v -I$(CUDA_BASE_DIR)/include/ $(CUDA_SAMPLES_INC) $(EXTRA_NVCC_FLAGS)
LINK_FLAGS = -std=c++11 -m64 -g $(GENCODE) -lcudart -lcudadevrt -lm $(EXTRA_LINK_FLAGS)

all: clean $(TARGET) install
#all: clean $(TARGET) 

$(TARGET): 
	$(NVCC) -ccbin g++ -o scalarProd.o -c scalarProd.cu $(NVCC_FLAGS)  -L$(CUDA_LIB_DIR) $(LINK_FLAGS)
	$(NVCC) -ccbin g++ -m64 $(GENCODE) $(CUDA_SAMPLES_INC) scalarProd_cpu.o -c scalarProd_cpu.cpp
	$(NVCC) $(NVCC_FLAGS) -o scalarProd scalarProd.o scalarProd_cpu.o $(LINK_FLAGS)

install:
	mkdir -p $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/
	cp $(TARGET) $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/

test:
	./scalarProd 

golden:
	./scalarProd >golden_stdout.txt 2>golden_stderr.txt

clean:
	rm -f *.o *~ scalarProd

clobber: clean
	rm -f sassifi-inst-counts.txt sassifi-scalar-insts.txt golden* *_regcount.p stdout* stderr*
