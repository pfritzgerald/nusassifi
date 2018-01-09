include ../env.mk
TARGET := dwt2d

# Files
CFILES := 
CXXFILES := 
CUFILES := main.cu dwt.cu components.cu dwt_cuda/fdwt53.cu dwt_cuda/fdwt97.cu dwt_cuda/common.cu dwt_cuda/rdwt97.cu dwt_cuda/rdwt53.cu

# Includes
INCLUDES := -I. -I$(CUDA_BASE_DIR)/include


# Common flags
COMMONFLAGS += $(INCLUDES) 
NVCC_FLAGS += $(COMMONFLAGS) -rdc=true -Xptxas -v $(EXTRA_NVCC_FLAGS) $(GENCODE)
CXXFLAGS += $(COMMONFLAGS)
CFLAGS += $(COMMONFLAGS) -std=c99 
LINK_FLAGS += -std=c++11 $(GENCODE) -lcudart -lcudadevrt $(EXTRA_LINK_FLAGS)

OUTPUT = -DOUTPUT

# Generate object files list
COBJS=$(CFILES:.c=.c.o)
CXXOBJS=$(CXXFILES:.cpp=.cpp.o)
CUOBJS=$(CUFILES:.cu=.cu.o)

.SUFFIXES: .c.o .cpp.o .cu.o .cu 

all: clean $(TARGET) install

%.c.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.cu.o: %.cu
	$(NVCC) -Xcompiler -fPIC  $(OUTPUT) $(NVCC_FLAGS) -c $< -o $@

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(COBJS) $(CXXOBJS) $(CUOBJS) 
	$(NVCC) -o $(TARGET) $(COBJS) $(CXXOBJS) $(CUOBJS) $(LINK_FLAGS)

install:
	mkdir -p $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/
	cp $(TARGET) $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/

test:
	./dwt2d 192.bmp -d 192x192 -f -5 -l 3

golden:
	./dwt2d 192.bmp -d 192x192 -f -5 -l 3 >golden_stdout.txt 2>golden_stderr.txt

clean:
	rm -f $(COBJS) $(CXXOBJS) $(CUOBJS) $(EXECUTABLE)

clobber: clean
	rm -f sassifi-inst-counts.txt golden*
	rm -f *.bmp.dwt.*
