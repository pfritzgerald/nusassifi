include ../env.mk
TARGET = srad_v1

NVCC_FLAGS = -O3 -c  -g -G -rdc=true -Xptxas -v -I$(CUDA_BASE_DIR)/include/ $(EXTRA_NVCC_FLAGS) $(GENCODE)
LINK_FLAGS = -std=c++11 $(GENCODE) -lcudart -lcudadevrt -lm $(EXTRA_LINK_FLAGS)


# Example
# target: dependencies
	# command 1
	# command 2
          # .
          # .
          # .
	# command n

# link objects(binaries) together

all: clean $(TARGET) install

$(TARGET):	main.o
	$(NVCC)	-o $(TARGET) main.o  -L$(CUDA_LIB_DIR) $(LINK_FLAGS) 

# compile main function file into object (binary)
main.o: 	main.cu \
				define.c \
				graphics.c \
				extract_kernel.cu \
				prepare_kernel.cu \
				reduce_kernel.cu \
				srad_kernel.cu \
				srad2_kernel.cu \
				compress_kernel.cu
	$(NVCC)	main.cu $(NVCC_FLAGS)
install:
	mkdir -p $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/
	cp $(TARGET) $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/

test:
	./srad_v1 100 0.5 502 458 ../data/srad/image.pgm
golden:
	./srad_v1 100 0.5 502 458 ../data/srad/image.pgm >golden_stdout.txt 2>golden_stderr.txt

# delete all object files
clean:
	rm -f *.o srad_v1
	
clobber: clean
	rm -f sassifi-inst-count.txt golden* image_out.pgm
