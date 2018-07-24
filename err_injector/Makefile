# Setup your variables in env.mk before building!
include ../../env.mk

# Decide which libraries you want to compile.
export LIBS := 	libprofiler.a \
		libbb_path_profiler.a \
		libbb_profiler.a \
		libbbv_profiler.a \
		libcfg_profiler.a \
		librfinjector.a \
		libinstinjector.a \
		libpath_profiler.a \
		libpcinstinjector.a \
		libpupc_profiler.a

NVCC = $(CUDA_HOME)/bin/nvcc
CUPTI = $(CUDA_HOME)/extras/CUPTI

ifndef INSTALL_DIR
	INSTALL_DIR := ../..
endif

INSTALLED_LIBS := $(addprefix $(INSTALL_DIR)/lib/, $(LIBS))

########################################
# Flags that are useful for debugging. 
########################################

# Instrumentation handler will be empty if EMPTY_HANDLER=1. 
EMPTY_HANDLER=0

# During error injection run, the error will not be injected if this
# DUMMY_INJECTION=1. The rest of the code will execute, including the functions
# that are written to inject the error. Just the c-code that perturbs the value
# will be turned off. 
DUMMY_INJECTION=0

# Print timing per kernel launch
TIMING=0

# Debug flags
INJ_DEBUG_LIGHT=0
INJ_DEBUG_HEAVY=0

# Flags used by the build commands below
RF_INJECTIONS=0
INST_INJECTIONS=1
INTERVAL_MODE_INJECTION=0

########################################
# build rules: try "make -j3"
########################################

install: $(INSTALLED_LIBS)

$(INSTALL_DIR)/lib/lib%.a: %.o
	ar r $@ $^
	ranlib $@

profiler.o: profiler.cu
	$(NVCC) -ccbin $(CCBIN) -std=c++11 \
		--compiler-options -Wall \
		-O3 $(GENCODE) \
		-lineinfo -c -rdc=true -o $@ $^ \
		--maxrregcount=16 \
		--compiler-options -Wall \
		-DEMPTY_HANDLER=$(EMPTY_HANDLER) \
		-DTIMING=$(TIMING) \
		-DINJ_DEBUG_LIGHT=$(INJ_DEBUG_LIGHT) \
		-DINJ_DEBUG_HEAVY=$(INJ_DEBUG_HEAVY) \
		-I$(CUPTI)/include/ \
		-I$(INSTALL_DIR)/utils/ \
		-I$(SASSI_HOME)/include/

%_profiler.o: %_profiler.cu
	$(NVCC) -ccbin $(CCBIN) -std=c++11 \
		--compiler-options -Wall   \
		-O3 $(GENCODE)             \
		-lineinfo -c -rdc=true -o $@ $^ \
		--maxrregcount=16 \
		--compiler-options -Wall \
		-DEMPTY_HANDLER=$(EMPTY_HANDLER) \
		-DTIMING=$(TIMING) \
		-DINJ_DEBUG_LIGHT=$(INJ_DEBUG_LIGHT) \
		-DINJ_DEBUG_HEAVY=$(INJ_DEBUG_HEAVY) \
		-I$(CUPTI)/include/ \
		-I$(INSTALL_DIR)/utils/ \
		-I$(SASSI_HOME)/include/ 

rfinjector.o: injector.cu 
	$(NVCC) -ccbin $(CCBIN) -std=c++11 \
		--compiler-options -Wall \
		-O3 $(GENCODE) \
		-lineinfo -c -rdc=true -o $@ $^ \
		--maxrregcount=16 \
		--compiler-options -Wall \
		-DEMPTY_HANDLER=$(EMPTY_HANDLER) \
		-DDUMMY_INJECTION=$(DUMMY_INJECTION) \
		-DTIMING=$(TIMING) \
		-DINJ_DEBUG_LIGHT=$(INJ_DEBUG_LIGHT) \
		-DINJ_DEBUG_HEAVY=$(INJ_DEBUG_HEAVY) \
		-DINJ_MODE=$(RF_INJECTIONS) \
		-DCC_DEBUG=$(CC_DEBUG) \
		-I$(CUPTI)/include/ \
		-I$(INSTALL_DIR)/utils/ \
		-I$(SASSI_HOME)/include/

instinjector.o: injector.cu
	$(NVCC) -ccbin $(CCBIN) -std=c++11 \
		--compiler-options -Wall \
		-O3 $(GENCODE) \
		-lineinfo -c -rdc=true -o $@ $^ \
		--maxrregcount=16 \
		--compiler-options -Wall \
		-DEMPTY_HANDLER=$(EMPTY_HANDLER) \
		-DDUMMY_INJECTION=$(DUMMY_INJECTION) \
		-DTIMING=$(TIMING) \
		-DINJ_DEBUG_LIGHT=$(INJ_DEBUG_LIGHT) \
		-DINJ_DEBUG_HEAVY=$(INJ_DEBUG_HEAVY) \
		-DINJ_MODE=$(INST_INJECTIONS) \
		-DINTERVAL_MODE_INJECTION=$(INTERVAL_MODE_INJECTION) \
		-DCC_DEBUG=$(CC_DEBUG) \
		-I$(CUPTI)/include/ \
		-I$(INSTALL_DIR)/utils/ \
		-I$(SASSI_HOME)/include/

pcinstinjector.o: pc_injector.cu
	$(NVCC) -ccbin $(CCBIN) -std=c++11 \
		--compiler-options -Wall \
		-O3 $(GENCODE) \
		-lineinfo -c -rdc=true -o $@ $^ \
		--maxrregcount=16 \
		--compiler-options -Wall \
		-DEMPTY_HANDLER=$(EMPTY_HANDLER) \
		-DDUMMY_INJECTION=$(DUMMY_INJECTION) \
		-DTIMING=$(TIMING) \
		-DINJ_DEBUG_LIGHT=$(INJ_DEBUG_LIGHT) \
		-DINJ_DEBUG_HEAVY=$(INJ_DEBUG_HEAVY) \
		-DINJ_MODE=$(INST_INJECTIONS) \
		-DINTERVAL_MODE_INJECTION=$(INTERVAL_MODE_INJECTION) \
		-DCC_DEBUG=$(CC_DEBUG) \
		-I$(CUPTI)/include/ \
		-I$(INSTALL_DIR)/utils/ \
		-I$(SASSI_HOME)/include/

clean:
	$(RM) $(INSTALLED_LIBS) *.o *.a *~ 
