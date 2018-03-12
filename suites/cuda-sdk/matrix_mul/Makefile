include ../env.mk
TARGET = matrix_mul

all: clean $(TARGET) install
#all: clean $(TARGET) 
NVCC_FLAGS = -O3 -g -rdc=true -Xptxas -v -I$(CUDA_BASE_DIR)/include/ $(CUDA_SAMPLES_INC) $(EXTRA_NVCCFLAGS)
LINK_FLAGS = -std=c++11 -m64 -g $(GENCODE) -lcudart -lcudadevrt -lm $(EXTRA_LDFLAGS)


$(TARGET): 
	$(NVCC) -ccbin g++ -o matrix_mul.o -c matrix_mul.cu $(NVCC_FLAGS) -L$(CUDA_LIB_DIR) $(LINK_FLAGS)
	$(NVCC) $(NVCC_FLAGS)  -L$(CUDA_LIB_DIR) -o matrix_mul matrix_mul.o $(LINK_FLAGS)

install:
	mkdir -p $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/
	cp $(TARGET) $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/

test:
	./matrix_mul 

golden:
	./matrix_mul >golden_stdout.txt 2>golden_stderr.txt

clean:
	rm -f *.o *~ matrix_mul

clobber: clean
	rm -f sassifi-inst-counts.txt sassifi-scalar-insts.txt golden* *_regcount.p stdout* stderr*
