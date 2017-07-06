# (c) 2007 The Board of Trustees of the University of Illinois.

# Default language wide options

# If you want to "intercept" main and exit, use this GCC-specific option.
# We only use this option for Fermi examples, where we don't have the luxury
# of using UVM.
LDWRAP := -Xlinker "--wrap=main" -Xlinker "--wrap=exit"

# SASSI specific
CUPTI_LIB_DIR = $(CUDA_PATH)/extras/CUPTI/lib64
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

ifeq ($(SASSI_OPTION),profiler)
SASSI_CUDACFLAGS = $(BEFORE_ALL) $(BEFORE_REG_MEM_INFO) $(GENCODE)
SASSI_CUDALDFLAGS = -L$(INST_LIB_DIR) -lprofiler $(CUPTI) 
endif

ifeq ($(SASSI_OPTION),inst_injector)
SASSI_CUDACFLAGS = $(AFTER_REG_MEM) $(AFTER_REG_MEM_INFO) $(BRANCH_AROUND) $(GENCODE)  -Xptxas --sassi-function-entry -Xptxas --sassi-bb-entry
SASSI_CUDALDFLAGS = -L$(INST_LIB_DIR) -linstinjector $(CUPTI) 
endif
 
ifeq ($(SASSI_OPTION),rf_injector)
SASSI_CUDACFLAGS = $(BEFORE_ALL) $(BEFORE_REG_MEM_INFO) $(GENCODE)
SASSI_CUDALDFLAGS = -L$(INST_LIB_DIR) -lrfinjector $(CUPTI)
endif

ifeq ($(SASSI_OPTION),cfg)
SASSI_CUDACFLAGS = -Xptxas --sassi-function-entry -Xptxas --sassi-bb-entry $(GENCODE)
SASSI_CUDALDFLAGS = -L$(INST_LIB_DIR) -lcfg $(CUPTI)
endif

ifeq ($(SASSI_OPTION),fritz_cfg)
SASSI_CUDACFLAGS = -Xptxas --sassi-function-entry -Xptxas --sassi-bb-entry $(GENCODE)
SASSI_CUDALDFLAGS = -L$(INST_LIB_DIR) -lfritz_cfg $(CUPTI)
endif

ifeq (${OPTION},fritz_profiler)
SASSI_CUDACFLAGS = $(LDWRAP) -lineinfo -Xptxas --sassi-function-entry -Xptxas --sassi-bb-entry $(GENCODE)
SASSI_CUDALDFLAGS = $(LDWRAP) -L$(INST_LIB_DIR) -lfritz_profiler $(CUPTI) -L$(BOOST_HOME)/lib -lboost_regex -lcrypto -Xlinker -rpath,$(BOOST_HOME)/lib
endif

# CUDA specific
LANG_CFLAGS=-I$(PARBOIL_ROOT)/common/include -I$(CUDA_PATH)/include
LANG_CXXFLAGS=$(LANG_CFLAGS)
LANG_LDFLAGS=-L$(CUDA_LIB_PATH)

LANG_CUDACFLAGS=$(LANG_CFLAGS)

CFLAGS=$(APP_CFLAGS) $(LANG_CFLAGS) $(PLATFORM_CFLAGS)
CXXFLAGS=$(APP_CXXFLAGS) $(LANG_CXXFLAGS) $(PLATFORM_CXXFLAGS)


CUDACFLAGS=-rdc=true -Xptxas -v $(LANG_CUDACFLAGS) $(PLATFORM_CUDACFLAGS) $(APP_CUDACFLAGS) $(SASSI_CUDACFLAGS)
CUDALDFLAGS= $(LANG_LDFLAGS) -std=c++11 $(GENCODE) -lcudart $(PLATFORM_CUDALDFLAGS)  $(APP_CUDALDFLAGS) $(SASSI_CUDALDFLAGS)

# Rules common to all makefiles

########################################
# Functions
########################################

# Add BUILDDIR as a prefix to each element of $1
INBUILDDIR=$(addprefix $(BUILDDIR)/,$(1))

# Add SRCDIR as a prefix to each element of $1
INSRCDIR=$(addprefix $(SRCDIR)/,$(1))


########################################
# Environment variable check
########################################

# The second-last directory in the $(BUILDDIR) path
# must have the name "build".  This reduces the risk of terrible
# accidents if paths are not set up correctly.
ifeq ("$(notdir $(BUILDDIR))", "")
$(error $$BUILDDIR is not set correctly)
endif

ifneq ("$(notdir $(patsubst %/,%,$(dir $(BUILDDIR))))", "build")
$(error $$BUILDDIR is not set correctly)
endif

.PHONY: run

ifeq ($(CUDA_PATH),)
FAILSAFE=no_cuda
else 
FAILSAFE=
endif

########################################
# Derived variables
########################################

ifeq ($(DEBUGGER),)
DEBUGGER=gdb
endif

OBJS = $(call INBUILDDIR,$(SRCDIR_OBJS))

########################################
# Rules
########################################

default: $(FAILSAFE) $(BUILDDIR) $(BIN)

run:
	@echo "Resolving CUDA runtime library..."
	@$(shell echo $(RUNTIME_ENV)) LD_LIBRARY_PATH="$(CUDA_LIB_PATH) $(CUPTI_LIB_DIR) ldd $(BIN) | grep cuda"
	@$(shell echo $(RUNTIME_ENV)) LD_LIBRARY_PATH="$(CUDA_LIB_PATH) $(CUPTI_LIB_DIR)  $(BIN) $(ARGS)"

debug:
	@echo "Resolving CUDA runtime library..."
	@$(shell echo $(RUNTIME_ENV)) LD_LIBRARY_PATH=$(CUDA_LIB_PATH) ldd $(BIN) | grep cuda
	@$(shell echo $(RUNTIME_ENV)) LD_LIBRARY_PATH=$(CUDA_LIB_PATH) $(DEBUGGER) --args $(BIN) $(ARGS)

clean :
	rm -rf $(BUILDDIR)/*
	if [ -d $(BUILDDIR) ]; then rmdir $(BUILDDIR); fi

$(BIN) : $(OBJS) $(BUILDDIR)/parboil_cuda.o
	$(CUDALINK) $^ -o $@ $(CUDALDFLAGS)

$(BUILDDIR) :
	mkdir -p $(BUILDDIR)

$(BUILDDIR)/%.o : $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/parboil_cuda.o: $(PARBOIL_ROOT)/common/src/parboil_cuda.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/%.o : $(SRCDIR)/%.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/%.o : $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/%.o : $(SRCDIR)/%.cu
	$(CUDACC) $< $(CUDACFLAGS) -c -o $@

no_cuda:
	@echo "CUDA_PATH is not set. Open $(CUDA_ROOT)/common/Makefile.conf to set default value."
	@echo "You may use $(PLATFORM_MK) if you want a platform specific configurations."
	@exit 1

