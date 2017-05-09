################################################################################### 
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###################################################################################


PYTHON_P = "python2.7"

TIMEOUT_THRESHOLD = 20 # 10X usual runtime 

#verbose = True
verbose = False

#######################################################################
# Categories of instruction types (IGIDs): This should match the values set in
# err_injector/error_injector.h. 
#######################################################################
IADD_IMUL_OP = 0
FADD_FMUL_OP = 1
MAD_OP = 2
FMA_OP = 3 
SETP_OP = 4 
LDS_OP = 5
LD_OP = 6
MISC_OP = 7

GPR = 8
CC = 9
PR = 10
STORE_VAL = 11
NUM_INST_TYPES = 12

IGID_STR = ["IADD_IMUL", "FADD_FMUL", "MAD",
"FMA", "SETP", "LDS", "LD",
"MISC_OP", "GPR", "CC", "PR", "STORE_VAL"]

#######################################################################
# Types of avaialble error models (bit-flip model, BFM): This should match the
# values set in err_injector/error_injector.h. 
#######################################################################
FLIP_SINGLE_BIT = 0
FLIP_TWO_BITS = 1
RANDOM_VALUE = 2
ZERO_VALUE = 3

WARP_FLIP_SINGLE_BIT = 4
WARP_FLIP_TWO_BITS = 5
WARP_RANDOM_VALUE = 6
WARP_ZERO_VALUE = 7

EM_STR = [ "FLIP_SINGLE_BIT", "FLIP_TWO_BITS", "RANDOM_VALUE", "ZERO_VALUE", 
"WARP_FLIP_SINGLE_BIT", "WARP_FLIP_TWO_BITS", "WARP_RANDOM_VALUE", "WARP_ZERO_VALUE"]

rf_inst = ""

#######################################################################
# Categories of error injection outcomes
#######################################################################
# Masked
MASKED_NOT_READ = 1
MASKED_WRITTEN = 2
MASKED_OTHER = 3

# DUEs
TIMEOUT = 4
NON_ZERO_EC = 5 # non zero exit code

# Potential DUEs with appropriate detectors in place
MASKED_KERNEL_ERROR = 6
SDC_KERNEL_ERROR = 7
NON_ZERO_EM = 8 # non zero error message (stderr is different)
STDOUT_ERROR_MESSAGE = 9
STDERR_ONLY_DIFF = 10
DMESG_STDERR_ONLY_DIFF = 11
DMESG_STDOUT_ONLY_DIFF = 12
DMESG_OUT_DIFF = 13
DMESG_APP_SPECIFIC_CHECK_FAIL= 14

# SDCs
STDOUT_ONLY_DIFF = 15
OUT_DIFF = 16
APP_SPECIFIC_CHECK_FAIL= 17

OTHERS = 18
NUM_CATS = 19

CAT_STR = ["Masked: Error was never read", "Masked: Write before read",
"Masked: other reasons", "DUE: Timeout", "DUE: Non Zero Exit Status", 
"Pot DUE: Masked but Kernel Error", "Pot DUE: SDC but Kernel Error", 
"Pot DUE: Different Error Message", "Pot DUE: Error Message in Standard Output", 
"Pot DUE: Stderr is different", "Pot DUE:Stderr is different, but dmesg recorded", 
"Pot DUE: Standard output is different, but dmesg recorded", 
"Pot DUE: Output file is different, but dmesg recorded", 
"Pot DUE: App specific check failed, but dmesg recorded",
"SDC: Standard output is different", "SDC: Output file is different", 
"SDC: App specific check failed", "Uncategorized"]

