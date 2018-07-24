include ../env.mk
TARGET = simple_add

#NVCC_FLAGS = -O3 -g -rdc=true -Xptxas -v
#LINK_FLAGS = -std=c++11 -g 
NVCC_FLAGS = -O3 -g -rdc=true -Xptxas -v -I$(CUDA_BASE_DIR)/include/ $(EXTRA_NVCC_FLAGS)
LINK_FLAGS = -std=c++11 -g $(GENCODE) -lcudart -lcudadevrt -lm $(EXTRA_LINK_FLAGS)

all: clean $(TARGET) install
#all: clean $(TARGET) 

$(TARGET): 
	$(NVCC) -o simple_add simple_add.cu $(NVCC_FLAGS) -L$(CUDA_LIB_DIR) $(LINK_FLAGS)

install:
	mkdir -p $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/
	cp $(TARGET) $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/

test:
	./simple_add 

golden:
	./simple_add >golden_stdout.txt 2>golden_stderr.txt

clean:
	rm -f *.o *~ simple_add

clobber: clean
	rm -f sassifi-inst-counts.txt sassifi-scalar-insts.txt golden* *_regcount.p stdout* stderr*
