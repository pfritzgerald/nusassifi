include ../env.mk

OUTPUT = -DOUTPUT

TARGET = heartwall
NVCC_FLAGS = -rdc=true -Xptxas -v -I./AVI -c -O3 -I$(CUDA_BASE_DIR)/include/ $(GENCODE) $(EXTRA_NVCC_FLAGS)
LINK_FLAGS = -std=c++11  $(GENCODE) -lcudart -lcudadevrt -lm $(EXTRA_LINK_FLAGS)
#

all: clean $(TARGET) install

# link objects(binaries) together
$(TARGET): main.o ./AVI/avilib.o ./AVI/avimod.o 
	$(NVCC) main.o ./AVI/avilib.o ./AVI/avimod.o $(LINK_FLAGS) -o $(TARGET)

# compile main function file into object (binary)
main.o: main.cu kernel.cu define.c
	$(NVCC) $(OUTPUT) $(KERNEL_DIM) main.cu $(NVCC_FLAGS)

./AVI/avilib.o ./AVI/avimod.o:
	cd AVI; make;

install:
	mkdir -p $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/
	cp $(TARGET) $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/

test:
	./heartwall ../data/heartwall/test.avi 20

golden:
	./heartwall ../data/heartwall/test.avi 20 >golden_stdout.txt 2>golden_stderr.txt



# delete all object files
clean:
	rm -f *.o AVI/*.o $(TARGET) *.linkinfo

clobber: clean
	rm -f sassifi-inst-counts.txt golden* result.txt
