BIN    		:= $(TOPLEVEL)/bin
INPUTS 		:= $(TOPLEVEL)/inputs

NVCC 		:= $(CUDA_BASE_DIR)/bin/nvcc
GCC  		:= g++
CC := $(GCC)
CUB_DIR := $(TOPLEVEL)/cub-1.6.4
SUITE_NAME := lonestargpu-2.0
OPTION=none

#----------------
LDWRAP := -Xlinker "--wrap=main" -Xlinker "--wrap=exit"

# SASSI specific
CUPTI_LIB_DIR = $(CUDA_BASE_DIR)/extras/CUPTI/lib64
CUPTI = -L$(CUPTI_LIB_DIR) -lcupti 

GENCODE = -gencode arch=compute_35,code=sm_35
AFTER_REG = -Xptxas --sassi-inst-after="reg-writes" 
AFTER_MEM = -Xptxas --sassi-inst-after="memory" 
AFTER_REG_MEM = -Xptxas --sassi-inst-after="reg-writes\,memory" 
BEFORE_ALL = -Xptxas --sassi-inst-before="all"
BEFORE_COND_BRANCHES = -Xptxas --sassi-inst-before="cond-branches"
BEFORE_MEM = -Xptxas --sassi-inst-before="memory"
BEFORE_REGS = -Xptxas --sassi-inst-before="reg-writes,reg-reads"

AFTER_REG_INFO = -Xptxas --sassi-after-args="reg-info"
AFTER_MEM_INFO = -Xptxas --sassi-after-args="mem-info"
AFTER_REG_MEM_INFO = -Xptxas --sassi-after-args="reg-info\,mem-info"
BEFORE_COND_BRANCH_INFO = -Xptxas --sassi-before-args="cond-branch-info"
BEFORE_MEM_INFO = -Xptxas --sassi-before-args="mem-info"
BEFORE_REG_INFO = -Xptxas --sassi-before-args="reg-info"
BEFORE_REG_MEM_INFO = -Xptxas --sassi-before-args="reg-info\,mem-info"

BRANCH_AROUND = -Xptxas --sassi-iff-true-predicate-handler-call
ifeq ($(OPTION), cubin)
SASSI_CUDACFLAGS = -cubin
endif

ifeq ($(OPTION),profiler)
SASSI_CUDACFLAGS = $(BEFORE_ALL) $(BEFORE_REG_MEM_INFO) $(GENCODE)
SASSI_CUDALDFLAGS = -L$(INST_LIB_DIR) -lprofiler $(CUPTI) 
endif

ifeq ($(OPTION),inst_injector)
SASSI_CUDACFLAGS = $(AFTER_REG_MEM) $(AFTER_REG_MEM_INFO) $(BRANCH_AROUND) $(GENCODE) 
SASSI_CUDALDFLAGS = -L$(INST_LIB_DIR) -linstinjector $(CUPTI) 
endif
 
ifeq ($(OPTION),rf_injector)
SASSI_CUDACFLAGS = $(BEFORE_ALL) $(BEFORE_REG_MEM_INFO) $(GENCODE)
SASSI_CUDALDFLAGS = -L$(INST_LIB_DIR) -lrfinjector $(CUPTI)
endif

ifeq ($(OPTION),cfg)
SASSI_CUDACFLAGS = -Xptxas --sassi-function-entry -Xptxas --sassi-bb-entry $(GENCODE)
SASSI_CUDALDFLAGS = -L$(INST_LIB_DIR) -lcfg $(CUPTI)
endif

ifeq ($(OPTION),fritz_cfg)
SASSI_CUDACFLAGS = -Xptxas --sassi-function-entry -Xptxas --sassi-bb-entry $(GENCODE)
SASSI_CUDALDFLAGS = -L$(INST_LIB_DIR) -lfritz_cfg $(CUPTI)
endif

ifeq (${OPTION},fritz_profiler)
SASSI_CUDACFLAGS = $(LDWRAP) -lineinfo -Xptxas --sassi-function-entry -Xptxas --sassi-bb-entry $(GENCODE)
SASSI_CUDALDFLAGS = $(LDWRAP) -L$(INST_LIB_DIR) -lfritz_profiler $(CUPTI) -L$(BOOST_HOME)/lib -lboost_regex -lcrypto -Xlinker -rpath,$(BOOST_HOME)/lib
endif

ifeq (${OPTION},memaccesses)
SASSI_CUDACFLAGS = $(BEFORE_ALL)  $(BEFORE_REG_MEM_INFO) -Xptxas --sassi-bb-entry#$(BEFORE_ALL) $(BEFORE_REG_MEM_INFO)
SASSI_CUDALDFLAGS = -L$(INST_LIB_DIR) -lmemaccesses $(CUPTI) 
endif

ifeq (${OPTION},bbv_profiler)
SASSI_CUDACFLAGS = $(AFTER_REG_MEM) $(AFTER_REG_MEM_INFO) -Xptxas --sassi-bb-entry $(BRANCH_AROUND) $(GENCODE)
SASSI_CUDALDFLAGS = -L$(INST_LIB_DIR) -lbbv_profiler $(CUPTI) 
endif

ifeq (${OPTION},mem_profiler)
SASSI_CUDACFLAGS = $(AFTER_REG_MEM)  $(AFTER_REG_MEM_INFO)  $(BRANCH_AROUND) $(GENCODE)
SASSI_CUDALDFLAGS = -L$(INST_LIB_DIR) -lmem_profiler $(CUPTI) 
endif

ifeq (${OPTION}, pupc_profiler)
SASSI_CUDACFLAGS = $(AFTER_REG_MEM)  $(AFTER_REG_INFO) $(BRANCH_AROUND) $(GENCODE)
SASSI_CUDALDFLAGS = -L$(INST_LIB_DIR) -lpupc_profiler $(CUPTI) 
endif

ifeq (${OPTION},bb_profiler)
SASSI_CUDACFLAGS = -Xptxas --sassi-function-entry -Xptxas --sassi-bb-entry
SASSI_CUDALDFLAGS = -L$(INST_LIB_DIR) -lbb_profiler $(CUPTI)
endif


#----------------------------

COMPUTECAPABILITY := sm_35
ifdef debug
FLAGS := -arch=$(COMPUTECAPABILITY) -g -DLSGDEBUG=1 -G
else
# including -lineinfo -G causes launches to fail because of lack of resources, pity.
FLAGS := -O3 -arch=$(COMPUTECAPABILITY) -g -Xptxas -v -rdc=true -std=c++11 -lcudart #-lineinfo -G
endif
INCLUDES := -I $(TOPLEVEL)/include -I $(CUB_DIR)
LINKS := 

EXTRA := $(FLAGS) $(INCLUDES) $(SASSI_CUDACFLAGS) $(LINKS) $(SASSI_CUDALDFLAGS)

all: clean $(APP) install

.PHONY: clean variants support optional-variants

ifdef APP
$(APP): $(SRC) $(INC)
	$(NVCC) $(EXTRA) -DVARIANT=0 -o $@ $<
	cp $@ $(BIN)

variants: $(VARIANTS)

optional-variants: $(OPTIONAL_VARIANTS)

support: $(SUPPORT)

install:
	mkdir -p $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)
	cp $(APP) $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/

test:
	./run

golden:
	./run > golden_stdout.txt 2>golden_stderr.txt
	
clean: 
	rm -f $(APP) $(BIN)/$(APP)
ifdef VARIANTS
	rm -f $(VARIANTS)
endif
ifdef OPTIONAL_VARIANTS
	rm -f $(OPTIONAL_VARIANTS)
endif

clobber: clean
	rm -f sassifi-inst-counts.txt golden* sassi-* 
endif
