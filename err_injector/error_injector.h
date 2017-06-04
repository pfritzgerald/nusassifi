/*********************************************************************************** \
* Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*  * Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*  * Neither the name of NVIDIA CORPORATION nor the names of its
*    contributors may be used to endorse or promote products derived
*    from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
* PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
\***********************************************************************************/

#ifndef __SASSIFI_STRUCTS___
#define __SASSIFI_STRUCTS___

#include <set>
#include <string>
#include "sassi/sassi-opcodes.h"

// configuration parameters 
#define NUM_INJ_PARAMS 5
#define MAX_REGS_PERINST 5
#define NUM_INJECTIONS 2

// Injection mode 
#define RF_INJECTIONS 0
#define INST_INJECTIONS 1

// Kernel name should be less than 200 characters
#define MAX_KNAME_SIZE 200

#define DEBUG_PRINT(FLAG, fmt, ...) 			\
	if (FLAG) printf(fmt, __VA_ARGS__);

//////////////////////////////////////////////////////////////////////
// Error model: Instruction grouping and bit-flip model 
//////////////////////////////////////////////////////////////////////

// List of the Bit Flip Models (BFMs)
enum BFM {
	FLIP_SINGLE_BIT = 0, 
	FLIP_TWO_BITS, // flip two adjacent bits
	RANDOM_VALUE,  // no random value error for CC, PR
	ZERO_VALUE,
	// The following models inject the same pattern in destination registers in
	// all the active threads in a warp 
	WARP_FLIP_SINGLE_BIT, 
	WARP_FLIP_TWO_BITS, 
	WARP_RANDOM_VALUE, 
	WARP_ZERO_VALUE, 
	NUM_BFM_TYPES 
};

// Arch state error injection sites instruction group IDs
// SASSIFI-user-guide summarizes these groups
enum INST_TYPE {
	IADD_IMUL_OP = 0, 
	FADD_FMUL_OP, 
	MAD_OP, 
	FMA_OP, 
	SETP_OP, 
	LDS_OP, 
	LD_OP, 
	MISC_OP, 
	GPR, 
	CC, 
	PR, 
	STORE_VAL, 
	NUM_INST_TYPES
};

const char *instCatName[NUM_INST_TYPES] = {
	"IADD_IMUL_OP", "FADD_FMUL_OP", "MAD_OP", 
	"FMA_OP", "SETP_OP", "LDS_OP", "LD_OP",
	"MISC_OP", "GPR", "CC", "PR", "STORE_VAL"
};

// Instruction groupings
// Input: opcode
// Output: the instruction group the opcode belongs to 
__device__ int get_op_category (int op) {
	switch (op) {
		 case SASSI_OP_IMUL:
		 case SASSI_OP_IMUL32I:
		 case SASSI_OP_ISCADD:
		 case SASSI_OP_ISCADD32I:
		 case SASSI_OP_IADD:
		 case SASSI_OP_IADD3:
		 case SASSI_OP_IADD32I: return IADD_IMUL_OP;

		 case SASSI_OP_DADD:
		 case SASSI_OP_DMUL:
		 case SASSI_OP_FADD:
		 case SASSI_OP_FMUL:
		 case SASSI_OP_FMUL32I:
		 case SASSI_OP_FADD32I: return FADD_FMUL_OP;

		 case SASSI_OP_XMAD:
		 case SASSI_OP_IMAD:
		 case SASSI_OP_IMAD32I:
		 case SASSI_OP_IMADSP: return MAD_OP;

		 case SASSI_OP_DFMA:
		 case SASSI_OP_FFMA:
		 case SASSI_OP_FFMA32I: return FMA_OP;

		 case SASSI_OP_PSETP:
		 case SASSI_OP_ISETP:
		 case SASSI_OP_FSETP:
		 case SASSI_OP_DSETP:
		 case SASSI_OP_CSETP: return SETP_OP;

		 case SASSI_OP_LDSLK:
		 case SASSI_OP_LDS_LDU:
		 case SASSI_OP_LDS: return LDS_OP;

		 case SASSI_OP_LDG:
		 case SASSI_OP_LDL:
		 case SASSI_OP_LDLK:
		 case SASSI_OP_LDU:
		 case SASSI_OP_LD_LDU:
		 case SASSI_OP_LD: return LD_OP;

		 default:  return MISC_OP; 
	}
}

//////////////////////////////////////////////////////////////////////
// Input/output filenames
//////////////////////////////////////////////////////////////////////

// output profile filename: This file is created by the profiler (profiler.cu)
std::string profileFilename = "sassifi-inst-counts.txt";

// injection parameters input filename: This file is created the the script
// that launched error injections
std::string injInputFilename = "sassifi-injection-seeds.txt";

//////////////////////////////////////////////////////////////////////
// Data structures for recording/storing infomration
//////////////////////////////////////////////////////////////////////

// Data structure to store 128-bit values
typedef struct uint128 {
	uint64_t values[2];
} uint128_t;

// Data structure to record memory infomration 
typedef struct mem_info {
	int64_t address;
	int32_t bitwidth;
	uint128_t value;
} mem_info_t;

__managed__ mem_info_t mem_info_d;


__device__ uint64_t get_recorded_memory_value() {
	return mem_info_d.value.values[0]; 
}

__device__ uint128_t get_recorded_memory_value128() {
	return mem_info_d.value;
}

// record memory value based on the bidwidth
__device__ void record_memory_info(int64_t address, int32_t bitwidth) {
	mem_info_d.bitwidth = bitwidth; // GetWidth returns bytes 
	mem_info_d.address = address; 

	mem_info_d.value.values[0] = 0;
	mem_info_d.value.values[1] = 0;

	if (bitwidth == 32) { // most common case
		mem_info_d.value.values[0] = (uint64_t) *((uint32_t*)address);
		DEBUG_PRINT(INJ_DEBUG_HEAVY, "memAddr=%llx: value before=%llx\n", address, mem_info_d.value.values[0]);
	} else if (bitwidth == 8) { 
		mem_info_d.value.values[0] = (uint64_t) *((uint8_t*)address);
	} else if (bitwidth == 16) { 
		mem_info_d.value.values[0] = (uint64_t) *((uint16_t*)address);
	} else if (bitwidth == 64) {
		mem_info_d.value.values[0] = *((uint64_t*)address);
	} else if (bitwidth == 128) {
		mem_info_d.value.values[0] =	(*((uint128_t*)address)).values[0];
		mem_info_d.value.values[1] =	(*((uint128_t*)address)).values[1];
	}
}

//////////////////////////////////////////////////////////////////////
// General accessor functions 
//////////////////////////////////////////////////////////////////////

// returns an integer idex of the destination using 
// inputs, injection seed (opID) and number of operands (numOps)
__device__ int32_t get_int_inj_id(int32_t numOps, float opID) {
	return (int32_t)(numOps*opID);
}

__device__ inline  bool is_store_inst(SASSICoreParams* ap, SASSIMemoryParams *mp) {
	if (ap->IsMem()) {// is memory instruction?
		if (mp->IsStore()) {
			return true;
		} 
	}
	return false;
}
__device__ inline bool is_load_inst(SASSICoreParams* ap, SASSIMemoryParams *mp) {
	if (ap->IsMem()) { // is memory instruction?
		if (mp->IsLoad()) {
			return true;
		}
	}
	return false;
}

// returns true if the instruction has a General Purpose Register (GPR) as one of the destinations
__device__ inline bool has_dest_GPR(SASSIRegisterParams *rp) {
	return (rp->GetNumGPRDsts() > 0);
}

// returns true if the instruction writes to Condition Code (CC) register
__device__ inline bool has_dest_CC(SASSIRegisterParams *rp)  {
	return rp->IsCCDefined(); 
}

// returns true if the instruction writes to a Predicate register (PR)
__device__ inline bool has_dest_PR(SASSIRegisterParams *rp)  {
	return (rp->GetPredicateDstMask() != 0); 
}

// returns true if the instruction has either GPR, CC, or PR as a destination register
__device__ bool has_dest_reg(SASSIRegisterParams *rp) {
	return (has_dest_GPR(rp) | has_dest_PR(rp) | has_dest_CC(rp)); 
}

// flatten thread id
__device__ int get_flat_tid() {
  int tid_b = threadIdx.x + (blockDim.x * (threadIdx.y + (threadIdx.z * blockDim.y))); // thread id within a block
  int bid = blockIdx.x + (gridDim.x * (blockIdx.y + (blockIdx.z * gridDim.y))); // block id 
	int tid = tid_b + (bid * blockDim.x * blockDim.y * blockDim.z);
	return tid;
}

// return the format in which the instruction profile is printed in the sassifi-inst-counts.txt file 
std::string  get_profile_format() {
	std::string first_line = "kName:kernelCount";
	for (int i=0; i<NUM_INST_TYPES; i++) 
		first_line += ":" + (std::string)instCatName[i] ;
	first_line += ":opWillNotExecuteCount";
	for (int i=0; i<SASSI_NUM_OPCODES; i++)
		first_line += ":" + (std::string)SASSIInstrOpcodeStrings[i] ;
	return first_line + "\n";
}

///////////////////////////////////////////////////////////////////////////////////
// profiling related code
///////////////////////////////////////////////////////////////////////////////////

// Counters that we keep on the device.
__managed__ unsigned long long injCounterAllInsts; // single counter for all instructions
__managed__ unsigned long long injCountersInstType[NUM_INST_TYPES]; // count instruction based on different instruction types as well as whether they write to GPR, CC, PR or STORE_VAL
__managed__ unsigned long long opCounters[SASSI_NUM_OPCODES]; // count instruction based on opcode
__managed__ unsigned long long opWillNotExecuteCount; // counter for instructions that will not execute 

__device__ inline void profile_instructions(SASSICoreParams* cp, SASSIMemoryParams* mp, SASSIRegisterParams* rp) {

	atomicAdd(&injCountersInstType[GPR], (unsigned long long)has_dest_GPR(rp)); 
	atomicAdd(&injCountersInstType[CC], (unsigned long long)has_dest_CC(rp)); 
	atomicAdd(&injCountersInstType[PR], (unsigned long long)has_dest_PR(rp)); 
	atomicAdd(&injCountersInstType[STORE_VAL], (unsigned long long)is_store_inst(cp, mp)); 

	int op = cp->GetOpcode();
	atomicAdd(&opCounters[op], 1LL);

	atomicAdd(&injCountersInstType[get_op_category(op)], 1LL); 
}

__device__ inline void profile_will_not_execute_instructions() {
	atomicAdd(&opWillNotExecuteCount, 1LL);
}


void reset_profiling_counters() {
	injCounterAllInsts = 0; 
	opWillNotExecuteCount = 0;
	bzero(injCountersInstType, sizeof(injCountersInstType));
	bzero(opCounters, sizeof(opCounters));
}

#endif // __SASSIFI_STRUCTS___
