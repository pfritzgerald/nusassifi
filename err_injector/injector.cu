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


#define __STDC_FORMAT_MACROS
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <string>
#include <fstream>
#include <algorithm>
#include <map>
#include <sys/time.h>
#include <cupti.h>

#include <sassi/sassi-core.hpp>
#include <sassi/sassi-regs.hpp>
#include <sassi/sassi-memory.hpp>
#include <sassi/sassi-opcodes.h>

#include <sassi/sassi-function.hpp>

#include "sassi_intrinsics.h"
#include "sassi_dictionary.hpp"
#include "sassi_lazyallocator.hpp"

#include "error_injector.h"

// knameCount keeps track of kernel names and the number of invocations of the
// cuda kernels that executed so far during the application execution.
std::map<std::string, int> knameCount; 

#if TIMING
struct timeval start, end;
float mTotalTime = 0;
#endif
// 8Mb of space for CFG information.
#define POOLSIZE (1024 * 1024 * 8)
#define MAX_FN_STR_LEN 64

// Create a memory pool that we can populate on the device and read on the host.
static __managed__ uint8_t sassi_mempool[POOLSIZE];
static __managed__ int     sassi_mempool_cur;

// A structure to record a basic block.  We will perform a deep copy
// of SASSI's SASSIBasicBlockParams for each basic block.
struct BLOCK {
  int id;
  unsigned long long weight;
  bool isEntry;
  bool isExit;
  int numInstrs;
  int numSuccs;
  int succs[2];
};

// A structure to record a function's CFG.
struct CFG {
  char fnName[MAX_FN_STR_LEN];
  int numBlocks;
  BLOCK *blocks;
};

// A dictionary of CFGs.
static __managed__ sassi::dictionary<int64_t, CFG*> *sassi_cfg;

// A dictionary of basic blocks.
static __managed__ sassi::dictionary<int64_t, BLOCK*> *sassi_cfg_blocks;

///////////////////////////////////////////////////////////////////////////////////
///
///  Allocate data from the UVM mempool.
///
///////////////////////////////////////////////////////////////////////////////////
__device__ void *simple_malloc(size_t sz)
{
  int ptr = atomicAdd(&sassi_mempool_cur, sz);
  assert ((ptr + sz) <= POOLSIZE);
  return (void*) &(sassi_mempool[ptr]);
}

///////////////////////////////////////////////////////////////////////////////////
///
///  A simple string copy to copy from device memory to our UVM malloc'd region.
///
///////////////////////////////////////////////////////////////////////////////////
__device__ void simple_strncpy(char *dest, const char *src)
{
  int i;
  for (i = 0; i < MAX_FN_STR_LEN-1; i++) {
    char c = src[i];
    if (c == 0) break;
    dest[i] = c;
  }
  dest[i] = '\0';
}
//////////////////////////////////////////////////////////////////////
// Error injection parameters and related functions
//////////////////////////////////////////////////////////////////////

typedef struct {
	bool areParamsReady;
	bool errorInjected;
	bool writeBeforeRead;
	bool readyToInject;
	char injKernelName[MAX_KNAME_SIZE]; 
	long long injThreadID; // injection thread id
	int32_t injKCount;
	int32_t injIGID; // arch state id
	unsigned long long injInstID; // injection inst id
	float injOpSeed; // injection operand id seed (random number between 0-1)
	uint32_t injBFM; // error model 
	float injBIDSeed; // bit id seed (random number between 0-1)
} inj_info_t; 

__managed__ inj_info_t inj_info; 

void reset_inj_info() {
	inj_info.areParamsReady = false;
	inj_info.errorInjected = false;
	inj_info.writeBeforeRead = false;
	inj_info.readyToInject = false;
	inj_info.injThreadID = -1; 
	inj_info.injKernelName[0] = '\0';
	inj_info.injKCount = 0;
	inj_info.injIGID = 0; // arch state id 
	inj_info.injInstID = 0; // instruction id 
	inj_info.injOpSeed = 0; // destination id seed (float, 0-1)
	inj_info.injBIDSeed = 0; // bit location seed (float, 0-1)
	inj_info.injBFM = 0; // fault model: single bit flip, all bit flip, random value
}

// for debugging 
void print_inj_info() {
	printf("inj_kname=%s, inj_kcount=%d, ", inj_info.injKernelName, inj_info.injKCount);
	printf("inj_igid=%d, inj_fault_model=%d, inj_inst_id=%lld",
			inj_info.injIGID, inj_info.injBFM, inj_info.injInstID);
	printf("inj_destination_id=%f, inj_bit_location=%f \n", inj_info.injOpSeed, inj_info.injBIDSeed);
}

// Parse error injection site info from a file. This should be done on host side.
void parse_params(std::string filename) {
	reset_inj_info(); 

 	std::ifstream ifs (filename.c_str(), std::ifstream::in);
	if (ifs.is_open()) {
#if INJ_MODE != RF_INJECTIONS
		ifs >> inj_info.injIGID; // arch state id 
		assert(inj_info.injIGID >=0 && inj_info.injIGID < NUM_INST_TYPES); // ensure that the value is in the expected range
#endif

		ifs >> inj_info.injBFM; // fault model: single bit flip, all bit flip, random value
		assert(inj_info.injBFM < NUM_BFM_TYPES); // ensure that the value is in the expected range

		ifs >> inj_info.injKernelName;
		ifs >> inj_info.injKCount;
		ifs >> inj_info.injInstID; // instruction id

		ifs >> inj_info.injOpSeed; // destination id seed (float, 0-1 for inst injections and 0-256 for reg)
#if INJ_MODE != RF_INJECTIONS
		assert(inj_info.injOpSeed >=0 && inj_info.injOpSeed < 1.01); // ensure that the value is in the expected range
#else
		assert(inj_info.injOpSeed >=0 && inj_info.injOpSeed < 257); // ensure that the value is in the expected range
#endif

		ifs >> inj_info.injBIDSeed; // bit location seed (float, 0-1)
		assert(inj_info.injBIDSeed >= 0 && inj_info.injBIDSeed < 1.01); // ensure that the value is in the expected range
	}
	ifs.close();

	if (INJ_DEBUG_LIGHT) {
		print_inj_info();
	}
}

//////////////////////////////////////////////////////////////////////
// Functions for actual error injection
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// Input: inMask and injBIDSeed
// Output: outMask that selects which bit to flip among the bits that are 1 in inMask
//////////////////////////////////////////////////////////////////////
__device__ uint8_t get_inj_mask(uint8_t inMask, uint8_t injBIDSeed) {
	uint8_t outMask = 1;
	uint8_t tempInMask = inMask;

	int i, count=0;
	for (i=0; i<8; i++) { // counting number of 1s in inMask
		if (tempInMask & 0x1  == 1) {
			count++;
		}
		tempInMask = tempInMask >> 1;
	}

	if (INJ_DEBUG_HEAVY) {
		printf(" count = %d \n", count);
	}

	uint8_t injBID = get_int_inj_id(count, injBIDSeed);

	if (INJ_DEBUG_HEAVY) {
		printf(" injBID = %d \n", injBID);
	}

	count = 0;
	tempInMask = inMask;
	for (i=0; i<8; i++) { // counting number of 1s in inMask
		if (tempInMask & 0x1  == 1) {
			if (count == injBID) 
				break;
			count++;
		}
		tempInMask = tempInMask >> 1;
		outMask = outMask << 1;
	}

	if (INJ_DEBUG_HEAVY) {
		printf(" inMask=%x, outMask=%x \n", inMask, outMask);
	}

	return outMask;
}


////////////////////////////////////////////////////////////////////////////////////
// Injecting errors in store instructions
////////////////////////////////////////////////////////////////////////////////////
template <typename T>
__device__ void inject_store_error_t(SASSIAfterParams* ap, SASSIMemoryParams *mp, float injBIDSeed, unsigned long long injInstID, int32_t bitwidth, uint32_t injBFM) {
	uint32_t injBID = get_int_inj_id(bitwidth, injBIDSeed);
	int64_t addr = mp->GetAddress(); 
	T *memAddr = (T*) addr;

	DEBUG_PRINT(INJ_DEBUG_LIGHT, "memAddr=%llx: value before=%llx\n", memAddr, *memAddr);
	printf(":::Injecting: pc=%llx bbId=%d opcode=%s tid=%d instCount=%lld instType=st%d injBID=%d:::", ap->GetPUPC(), ap->GetBBID(), SASSIInstrOpcodeStrings[ap->GetOpcode()], get_flat_tid(), injInstID, bitwidth, injBID);

	if (!DUMMY_INJECTION) {
		if(injBFM == FLIP_SINGLE_BIT || injBFM == WARP_FLIP_SINGLE_BIT) {
			*memAddr = *memAddr ^ ((T)1<<injBID); // actual error injection
		} else if (injBFM == FLIP_TWO_BITS || injBFM == WARP_FLIP_TWO_BITS) {
			*memAddr = *memAddr ^ ((T)3<<injBID); // actual error injection
		} else if (injBFM == RANDOM_VALUE || injBFM == WARP_RANDOM_VALUE) { 
			*memAddr = ((T)(-1))*injBIDSeed; 
		} else if (injBFM == ZERO_VALUE || injBFM == WARP_ZERO_VALUE) {
			*memAddr = 0; 
		}
	}

	DEBUG_PRINT(INJ_DEBUG_LIGHT, "memAddr=%llx: value after=%llx\n", memAddr, *memAddr);
}

__device__ void inject_store128_error_t(SASSIAfterParams* ap, SASSIMemoryParams *mp, float injBIDSeed, unsigned long long injInstID, int32_t bitwidth, uint32_t injBFM) {
	uint32_t injBID = get_int_inj_id(bitwidth, injBIDSeed);
	int64_t addr = mp->GetAddress(); 
	uint128_t *memAddr = (uint128_t*) addr;

	DEBUG_PRINT(INJ_DEBUG_LIGHT, "memAddr=%llx: value before=%llx, %llx\n", memAddr, (*memAddr).values[0], (*memAddr).values[1]);
	printf(":::Injecting: pc=%llx bbId=%d opcode=%s tid=%d instCount=%lld instType=st%d injBID=%d:::", ap->GetPUPC(), ap->GetBBID(), SASSIInstrOpcodeStrings[ap->GetOpcode()], get_flat_tid(), injInstID, bitwidth, injBID);

	if (!DUMMY_INJECTION) {
		if (injBFM == FLIP_SINGLE_BIT || injBFM == WARP_FLIP_SINGLE_BIT) {
			if (injBID < 64) {
				memAddr->values[0] = memAddr->values[0] ^ ((uint64_t)1<<injBID); // actual error injection
			} else {
				memAddr->values[1] = memAddr->values[1] ^ ((uint64_t)1<<(injBID-64)); // actual error injection
			}
		} else if (injBFM == FLIP_TWO_BITS || injBFM == WARP_FLIP_TWO_BITS) {
			if (injBID < 63) {
				memAddr->values[0] = memAddr->values[0] ^ ((uint64_t)3<<injBID); // actual error injection
			} else if (injBID == 63) {
				memAddr->values[0] = memAddr->values[0] ^ ((uint64_t)1<<injBID); // actual error injection
				memAddr->values[1] = memAddr->values[1] ^ ((uint64_t)1<<(injBID-64)); // actual error injection
			} else {
				memAddr->values[1] = memAddr->values[1] ^ ((uint64_t)3<<(injBID-64)); // actual error injection
			}
		} else if (injBFM == RANDOM_VALUE || injBFM == WARP_RANDOM_VALUE) { 
			memAddr->values[0] = ((uint64_t)(-1))*injBIDSeed; 
			memAddr->values[1] = ((uint64_t)(-1))*injBIDSeed; 
		} else if (injBFM == ZERO_VALUE || injBFM == WARP_ZERO_VALUE) {
			memAddr->values[0] = 0; 
			memAddr->values[1] = 0; 
		}
	}

	DEBUG_PRINT(INJ_DEBUG_LIGHT, "memAddr=%llx: value before=%llx, %llx\n", memAddr, memAddr->values[0], memAddr->values[1]);
}

// Inject in store value
__device__ void inject_store_error(SASSIAfterParams* ap, SASSIMemoryParams *mp, float injBIDSeed, unsigned long long injInstID, uint32_t injBFM) {

	int32_t bitwidth = 8*mp->GetWidth(); // GetWidth returns bytes 
	if (bitwidth == 32) { // most common case
		inject_store_error_t<uint32_t>(ap, mp, injBIDSeed, injInstID, bitwidth, injBFM);
	} else if (bitwidth == 8) { 
		inject_store_error_t<uint8_t>(ap, mp, injBIDSeed, injInstID, bitwidth, injBFM);
	} else if (bitwidth == 16) { 
		inject_store_error_t<uint16_t>(ap, mp, injBIDSeed, injInstID, bitwidth, injBFM);
	} else if (bitwidth == 64) {
		inject_store_error_t<uint64_t>(ap, mp, injBIDSeed, injInstID, bitwidth, injBFM);
	} else if (bitwidth == 128) {
		inject_store128_error_t(ap, mp, injBIDSeed, injInstID, bitwidth, injBFM);
		DEBUG_PRINT(1, "WARNING: No injection for bitwidth=%d\n", bitwidth);
	}
}


////////////////////////////////////////////////////////////////////////////////////
// Injecting errors in GPR registers 
////////////////////////////////////////////////////////////////////////////////////
__device__ void inject_GPR_error(SASSICoreParams* cp, SASSIRegisterParams *rp, SASSIRegisterParams::GPRRegInfo regInfo, float injBIDSeed, unsigned long long injInstID, uint32_t injBFM) {

	// get the value in the register, and inject error
	int32_t valueInReg = rp->GetRegValue(cp, regInfo).asInt; 

	DEBUG_PRINT(INJ_DEBUG_LIGHT, "Injection candidate: register destination = %d \n", rp->GetRegNum(regInfo));
	DEBUG_PRINT(INJ_DEBUG_LIGHT, "Before Injection: register value = %x \n", valueInReg);


	SASSIRegisterParams::GPRRegValue injectedVal;
	injectedVal.asUint = 0;
	uint32_t injBID = 0;
	if (injBFM == FLIP_SINGLE_BIT || injBFM == WARP_FLIP_SINGLE_BIT) {
		injBID = get_int_inj_id(32, injBIDSeed);
		injectedVal.asUint = valueInReg ^ (1<<injBID); // actual error injection
	} else if (injBFM == FLIP_TWO_BITS || injBFM == WARP_FLIP_TWO_BITS) {
		injBID = get_int_inj_id(31, injBIDSeed);
		injectedVal.asUint = valueInReg ^ (3<<injBID); // actual error injection
	} else if (injBFM == RANDOM_VALUE || injBFM == WARP_RANDOM_VALUE) {
		injectedVal.asUint = ((uint32_t)-1) * injBIDSeed; 
	} else if (injBFM == ZERO_VALUE || injBFM == WARP_ZERO_VALUE) {
		injectedVal.asUint = 0; 
	}

	printf(":::Injecting: pc=%llx bbId=%d GlobalInstCount=%lld opcode=%s tid=%d instCount=%lld instType=GPR regNum=%d injBID=%d:::", 
			cp->GetPUPC(), cp->GetBBID(), injCounterAllInsts, SASSIInstrOpcodeStrings[cp->GetOpcode()], get_flat_tid(), injInstID,
			 rp->GetRegNum(regInfo), injBID);

	if (!DUMMY_INJECTION) {
		rp->SetRegValue(cp, regInfo, injectedVal); 
	}

	int32_t valueInRegAfter = rp->GetRegValue(cp, regInfo).asInt; 
	DEBUG_PRINT(INJ_DEBUG_LIGHT, "After Injection: register value = %x, ", valueInRegAfter);
	DEBUG_PRINT(INJ_DEBUG_LIGHT, "injectedVal = %x \n", injectedVal.asUint);
}

////////////////////////////////////////////////////////////////////////////////////
// Injecting errors in CC registers 
////////////////////////////////////////////////////////////////////////////////////
__device__ void inject_CC_error(SASSIAfterParams* ap, SASSIRegisterParams *rp, float injBIDSeed, unsigned long long injInstID, uint32_t injBFM) {

		uint8_t valueInReg = rp->SASSIGetCCRegisterVal(ap);  // read CC register value, only low 4 bits are used
		uint8_t injBID = get_int_inj_id(4, injBIDSeed);

		DEBUG_PRINT(INJ_DEBUG_LIGHT, "Before Injection: CC register value = %x \n", valueInReg);
		printf(":::Injecting: pc=%llx bbId=%d opcode=%s tid=%d instCount=%lld instType=CC injBID=%d:::", ap->GetPUPC(),
                    ap->GetBBID(), SASSIInstrOpcodeStrings[ap->GetOpcode()], get_flat_tid(), injInstID, injBID);
	
		uint8_t injectedVal = 0;
		if (injBFM == FLIP_SINGLE_BIT) {
			injectedVal = valueInReg ^ (1<<injBID); // actual error injection
		} 

		if (!DUMMY_INJECTION) {
			rp->SASSISetCCRegisterVal(ap, injectedVal); 
		}

		uint8_t valueInRegAfter  = rp->SASSIGetCCRegisterVal(ap); 
		DEBUG_PRINT(INJ_DEBUG_LIGHT, "After Injection: register value = %x ", valueInRegAfter);
		DEBUG_PRINT(INJ_DEBUG_LIGHT, ", injectedVal = %x \n", injectedVal);
}

////////////////////////////////////////////////////////////////////////////////////
// Injecting errors in PR registers 
////////////////////////////////////////////////////////////////////////////////////
__device__ void inject_PR_error(SASSIAfterParams* ap, SASSIRegisterParams *rp, float injBIDSeed, unsigned long long injInstID, uint32_t injBFM) {

	uint8_t valueInReg = rp->SASSIGetPredicateRegisterVal(ap);  // read PR register value 

	DEBUG_PRINT(INJ_DEBUG_LIGHT, "Before Injection: PR register value = %x \n", valueInReg);
	printf(":::Injecting: pc=%llx bbId=%d opcode=%s tid=%d instCount=%lld instType=PR injBID=0:::", ap->GetPUPC(), ap->GetBBID(), SASSIInstrOpcodeStrings[ap->GetOpcode()], get_flat_tid(), injInstID);

	uint8_t injectedVal = 0; 
	if (injBFM == FLIP_SINGLE_BIT) {
		injectedVal = valueInReg ^ get_inj_mask(rp->GetPredicateDstMask(), injBIDSeed); // actual error injection
	}

	if (!DUMMY_INJECTION) {
		rp->SASSISetPredicateRegisterVal(ap, injectedVal); 
	}

	uint8_t valueInRegAfter  = rp->SASSIGetPredicateRegisterVal(ap); 
	DEBUG_PRINT(INJ_DEBUG_LIGHT, "After Injection: register value = %x ", valueInRegAfter);
	DEBUG_PRINT(INJ_DEBUG_LIGHT, ", injectedVal = %x \n", injectedVal);
}

////////////////////////////////////////////////////////////////////////////////////
// Injecting errors in any destination register 
////////////////////////////////////////////////////////////////////////////////////
__device__ __noinline__ void inject_reg_error(SASSIAfterParams* ap, SASSIRegisterParams *rp, float injOpSeed, float injBIDSeed, unsigned long long injInstID, uint32_t injBFM) {

	int32_t numDestRegs = rp->GetNumGPRDsts(); // Get the number of destination registers assigned by this instruction.
	int32_t numDestOps = numDestRegs + rp->IsCCDefined() +  rp->GetPredicateDstMask() != 0; // num gpr regs + 1 for CC + 1 for PR
	DEBUG_PRINT(INJ_DEBUG_LIGHT, "At: tid=%d instCount=%lld opcode=%s numDestOps=%d, isCCDefined=%d, isPredicateDefined=%d\n", get_flat_tid(), injInstID, SASSIInstrOpcodeStrings[ap->GetOpcode()], numDestOps, rp->IsCCDefined(), rp->GetPredicateDstMask() != 0);

	if (numDestOps == 0)  // cannot inject - no destination operands 
		return;

	int32_t injOpID = get_int_inj_id(numDestOps, injOpSeed);
	if (injOpID < numDestRegs) { // inject in a GPR
		SASSIRegisterParams::GPRRegInfo regInfo = rp->GetGPRDst(injOpID); // get destination register info, get the value in that register, and inject error
		inject_GPR_error(ap, rp, regInfo, injBIDSeed, injInstID, injBFM);
	} else if (injOpID - numDestRegs + 1  == rp->IsCCDefined()) { // inject in CC register
		inject_CC_error(ap, rp, injBIDSeed, injInstID, injBFM);
	} else { // inject in PR Register
		inject_PR_error(ap, rp, injBIDSeed, injInstID, injBFM);
	} 
}

// return 0 if the injRegID is not found in the list of destination registers, else returns the index
__device__ int32_t is_dest_reg(SASSIRegisterParams *rp, int32_t injRegID) {
	int32_t numDestRegs = rp->GetNumGPRDsts(); // Get the number of destination registers assigned by this instruction.
	for (int32_t i=0; i<numDestRegs; i++) {
		if (rp->GetRegNum(rp->GetGPRDst(i)) == injRegID) {
			DEBUG_PRINT(INJ_DEBUG_LIGHT, "Injection candidate found in destination: register destination = %d \n", injRegID);
			return i;
		}
	}
	return 0;
}

// return 0 if the injRegID is not found in the list of source registers, else returns the index
__device__ int32_t is_src_reg(SASSIRegisterParams *rp, int32_t injRegID) {
	int32_t numSrcRegs = rp->GetNumGPRSrcs(); // Get the number of destination registers assigned by this instruction.
	for (int32_t i=0; i<numSrcRegs; i++) {
		if (rp->GetRegNum(rp->GetGPRSrc(i)) == injRegID) {
			DEBUG_PRINT(INJ_DEBUG_LIGHT, "Injection candidate found in source: register destination = %d \n", injRegID);
			return 1;
		}
	}
	return 0;
}

///////////////////////////////////////////////////////////////////////////////////
//  SASSI before handler: This function will be called before the instruction
//  gets executed. This is used only for RF-AVF injections. This function first
//  marks the register for injection. It then checks whether the register is
//  used in subsequent instructions. If it is used as a destination before
//  being read, the injection run terminates the run is categorized as masked.
//  If the register is not found in any of the source registers before the
//  thread exits, the injection run is categorized as masked. 
///////////////////////////////////////////////////////////////////////////////////
__device__ void sassi_before_handler(SASSIBeforeParams* bp, SASSIMemoryParams *mp, SASSIRegisterParams *rp) {
#if EMPTY_HANDLER && INJ_MODE != RF_INJECTIONS // if you don't want to inject RF based errors, return
	return;
#endif

	if (!inj_info.areParamsReady) 	// Check if this is the kernel of interest 
		return; 											// This is not the selected kernel. No need to proceed.

	unsigned long long currInstCounter = atomicAdd(&injCounterAllInsts, 1LL) + 1;  // update counter, returns old value
	if (inj_info.injInstID == currInstCounter) { // the current instruction count matches injInstID matches, time to inject the erorr
		// record thread number, and RF to inject.	
		// Note we are not injecting the error here, we are just recording it. We
		// will inject the error when it is used as a source register by the
		// subsequent instructions. 
		inj_info.injThreadID = get_flat_tid();
		inj_info.readyToInject = true;
		DEBUG_PRINT(INJ_DEBUG_LIGHT, "Injection point reached: tid=%lld instCount=%lld \n", inj_info.injThreadID, inj_info.injInstID);
	} 

	// if readyToInject is set and this is the thread that was selected, check for error injection
	if (inj_info.readyToInject && inj_info.injThreadID == get_flat_tid() && !inj_info.errorInjected) {
		// check if the selected register is either in source registers or destination registers
		if (is_dest_reg(rp, inj_info.injOpSeed) != 0) {
			DEBUG_PRINT(INJ_DEBUG_LIGHT, "Terminating run: Write found before a read tid=%lld\n", inj_info.injThreadID);
			// Record this injection as Masked and terminate
			inj_info.writeBeforeRead = true;
		  __threadfence();         // ensure store issued before trap
		  asm("trap;");            // kill kernel with error
		}
		int32_t src_reg = is_src_reg(rp, inj_info.injOpSeed);
		if (src_reg != 0) {
			DEBUG_PRINT(INJ_DEBUG_LIGHT, "Reached actual injection point tid=%lld\n", inj_info.injThreadID);
			inj_info.errorInjected = true;
			inject_GPR_error(bp, rp, rp->GetGPRSrc(src_reg), inj_info.injBIDSeed, inj_info.injInstID, inj_info.injBFM); // Inject the error and contine
		}
	}
	
}

///////////////////////////////////////////////////////////////////////////////////
//  SASSI After handler: This function is called after every SASS instruction.
//  This is used for instruction output-level injections only. This function
//  first checks whether the injection parameters are ready. If so, it checks
//  the instruction group id of the current instruction and then proceeds to
//  the respective function to check and perform error injection. 
///////////////////////////////////////////////////////////////////////////////////
__device__ void sassi_after_handler(SASSIAfterParams* ap, SASSIMemoryParams *mp, SASSIRegisterParams *rp) { // order is important

//	unsigned long long GlobalInstCounter = 

#if EMPTY_HANDLER && INJ_MODE != INST_INJECTIONS // if you don't want to inject instruction level errors, return
	return;
#endif

	if (!inj_info.areParamsReady) // Check if this is the kernel of interest 
		return; // This is not the selected kernel. No need to proceed.
 	atomicAdd(&injCounterAllInsts, 1LL) + 1;  // update counter, returns old value

	switch (inj_info.injIGID) {
		case GPR: {
				if (has_dest_GPR(rp)) {

					unsigned long long currInstCounter = atomicAdd(&injCountersInstType[GPR], 1LL); // update counter, return old value 
					bool cond = inj_info.injInstID == currInstCounter; // the current opcode matches injIGID and injInstID matches
					if (inj_info.injBFM == WARP_FLIP_SINGLE_BIT || inj_info.injBFM == WARP_FLIP_TWO_BITS  || inj_info.injBFM == WARP_RANDOM_VALUE || inj_info.injBFM == ZERO_VALUE || inj_info.injBFM == WARP_ZERO_VALUE) {  // For warp wide injections 
						cond = (__any(cond) != 0) ; // __any() evaluates cond for all active threads of the warp and return non-zero if and only if cond evaluates to non-zero for any of them.
					}
	
					if(cond) { 
						 // get destination register info, get the value in that register, and inject error
						SASSIRegisterParams::GPRRegInfo regInfo = rp->GetGPRDst(get_int_inj_id(rp->GetNumGPRDsts(), inj_info.injOpSeed));
						inject_GPR_error(ap, rp, regInfo, inj_info.injBIDSeed, inj_info.injInstID, inj_info.injBFM);
					}
				}
			}
			break;

		case CC: {
				if (has_dest_CC(rp)) {
					if (inj_info.injInstID == atomicAdd(&injCountersInstType[CC], 1LL)) {
						inject_CC_error(ap, rp, inj_info.injBIDSeed, inj_info.injInstID, inj_info.injBFM);
					}
				}
			}
			break;
		case PR: {
				if (has_dest_PR(rp)) {
					if (inj_info.injInstID == atomicAdd(&injCountersInstType[PR], 1LL)) {
						inject_PR_error(ap, rp, inj_info.injBIDSeed, inj_info.injInstID, inj_info.injBFM);
					}
				}
			}
			break;
		case STORE_VAL: {
				if (is_store_inst(ap, mp)) {
					unsigned long long currInstCounter = atomicAdd(&injCountersInstType[STORE_VAL], 1LL); // update counter, return old value 
					bool cond = inj_info.injInstID == currInstCounter; // the current opcode matches injIGID and injInstID matches
					if (inj_info.injBFM == WARP_FLIP_SINGLE_BIT || inj_info.injBFM == WARP_FLIP_TWO_BITS  || inj_info.injBFM == WARP_RANDOM_VALUE || inj_info.injBFM == ZERO_VALUE || inj_info.injBFM == WARP_ZERO_VALUE) {  // For warp wide injections 
						cond = (__any(cond) != 0) ; // __any() evaluates cond for all active threads of the warp and return non-zero if and only if cond evaluates to non-zero for any of them.
					}
	
					if(cond) { 
						inject_store_error(ap, mp, inj_info.injBIDSeed, inj_info.injInstID, inj_info.injBFM);
					}
				}
			}
			break;

		case LD_OP: 
		case LDS_OP: 
		case IADD_IMUL_OP: 
		case FADD_FMUL_OP: 
		case MAD_OP: 
		case FMA_OP: 
		case SETP_OP: {
				int32_t currInstCat = get_op_category(ap->GetOpcode());
				unsigned long long currInstCounter = atomicAdd(&injCountersInstType[currInstCat], 1LL);  // update counter, return old value 
			
				bool cond = inj_info.injIGID == currInstCat && inj_info.injInstID == currInstCounter; // the current opcode matches injIGID and injInstID matches
				
				if (inj_info.injBFM == WARP_FLIP_SINGLE_BIT || inj_info.injBFM == WARP_FLIP_TWO_BITS  || inj_info.injBFM == WARP_RANDOM_VALUE || inj_info.injBFM == ZERO_VALUE || inj_info.injBFM == WARP_ZERO_VALUE) {  // For warp wide injections 
					cond = (__any(cond) != 0) ; // __any() evaluates cond for all active threads of the warp and return non-zero if and only if cond evaluates to non-zero for any of them.
				}
	
				if(cond) { 
					inject_reg_error(ap, rp, inj_info.injOpSeed, inj_info.injBIDSeed, inj_info.injInstID, inj_info.injBFM); 
				}
			}
			break;
		case MISC_OP:  break;
	}
}

//////////////////////////////////////////////////////////////////////
// SASSI initialize, finalize, and other operations to be performed
// on kernel entry and exit 
//////////////////////////////////////////////////////////////////////

static void sassi_init() {
	 // read seeds for random error injection
	parse_params(injInputFilename.c_str());  // injParams are updated based on injection seed file
	sassi_mempool_cur = 0;
	bzero(sassi_mempool, sizeof(sassi_mempool));
	sassi_cfg = new sassi::dictionary<int64_t, CFG*>(601);
	sassi_cfg_blocks = new sassi::dictionary<int64_t, BLOCK*>(7919);
}

static void take_checkpoint()
{
	cudaDeviceSynchronize();
	FILE *cfgFile = fopen("checkpoint", "w");
	sassi_cfg->map([cfgFile](int64_t k, CFG* &cfg) {
	fprintf(cfgFile, "//CHECKPOINT: %s: LastInvocation: %d\n", 
			cfg->fnName, knameCount[cfg->fnName]);
	for (int bb = 0; bb < cfg->numBlocks; bb++) {
	BLOCK *block = &(cfg->blocks[bb]);

	fprintf(cfgFile, "\tBB=%d weight=%lld, ins=%d\n", 
		block->id, block->weight, block->numInstrs);
	}
	for (int bb = 0; bb < cfg->numBlocks; bb++) {
		BLOCK *block = &(cfg->blocks[bb]);
		for (int s = 0; s < block->numSuccs; s++) {
		fprintf(cfgFile, "\tBB%d -> BB%d;\n", block->id, block->succs[s]);
		}
	}
	});
	fclose(cfgFile);
}


//////////////////////////////////////////////////////////////////////
// This function is invoked before a cuda-kernel starts executing. 
// It resets profiling counters, updated knameCount (to keep track of how many
// kernels and their invocations are done), and updates injection parameters
// that are used by SASSI before and after handlers. 
//////////////////////////////////////////////////////////////////////
static void onKernelEntry(const CUpti_CallbackData *cbInfo) {

	reset_profiling_counters();

	// update knameCount map 
	std::string currKernelName = cbInfo->symbolName;
	if (knameCount.find(currKernelName) == knameCount.end()) {
		knameCount[currKernelName] = 0;
	} else {
		knameCount[currKernelName] += 1;
	}

	std::string injKernelName = inj_info.injKernelName;
 	// pass injParams if this is not the kernel of interest
	bool is_inj_kernel_name = injKernelName.compare(cbInfo->symbolName) == 0; // if the current kernel name is not same as injKernelName
	bool is_inj_kernel_count = (knameCount.find(injKernelName) != knameCount.end()) ? knameCount[injKernelName] == inj_info.injKCount : false; // if kernel name is found, check if injKCount matches knameCount[injKernelName]

	inj_info.areParamsReady = is_inj_kernel_name && is_inj_kernel_count; // mark the injection params ready

	if (inj_info.areParamsReady) 
		DEBUG_PRINT(INJ_DEBUG_LIGHT, "areParamsReady=%d, injkname=%s, curr kname=%s, injKCount=%d, is_inj_kernel_count=%d \n", inj_info.areParamsReady, injKernelName.c_str(), cbInfo->symbolName, inj_info.injKCount, is_inj_kernel_count);

#if TIMING 
	gettimeofday(&start, NULL);
#endif

  take_checkpoint();
} 
///////////////////////////////////////////////////////////////////////////////////
///
///  A call to this function will be inserted at the beginning of every 
///  CUDA function or kernel.  We will essentially perform a deep copy of the
///  CFG SASSI presents.  All of the copied data only contains static information
///  about the CFG.  In the sassi_basic_block_entry handler, below, we will 
///  record the dynamic number of times the basic block is invoked.
///
///////////////////////////////////////////////////////////////////////////////////
__device__ void sassi_function_entry(SASSIFunctionParams* fp)
{
  int numBlocks = fp->GetNumBlocks();
  const SASSIBasicBlockParams * const * blocks = fp->GetBlocks();
  
  CFG *cPtr = *(sassi_cfg->getOrInit((int64_t)fp, [numBlocks, blocks, fp](CFG **cfg) {
      CFG *cPtr = (CFG*) simple_malloc(sizeof(CFG));
      simple_strncpy(cPtr->fnName, fp->GetFnName());
      cPtr->numBlocks = numBlocks;
      cPtr->blocks = (BLOCK*) simple_malloc(sizeof(BLOCK) * numBlocks);
      *cfg = cPtr;
  }));

  __threadfence();

  for (int bb = 0; bb < numBlocks; bb++) {
    const SASSIBasicBlockParams *blockParam = blocks[bb];
    BLOCK *blockPtr = &(cPtr->blocks[bb]);    
    sassi_cfg_blocks->getOrInit((int64_t)blockParam, [blockParam, blockPtr](BLOCK **bpp) {
	*bpp = blockPtr;
	blockPtr->id = blockParam->GetID();
	blockPtr->weight = 0;
	blockPtr->isEntry = blockParam->IsEntryBlock();
	blockPtr->isExit = blockParam->IsExitBlock();
	blockPtr->numInstrs = blockParam->GetNumInstrs();
	blockPtr->numSuccs = blockParam->GetNumSuccs();
	assert(blockParam->GetNumSuccs() <= 2);
	const SASSIBasicBlockParams * const * succs = blockParam->GetSuccs();
	for (int s = 0; s < blockParam->GetNumSuccs(); s++) {
	  blockPtr->succs[s] = succs[s]->GetID();
	}
      });
  }
}
//////////////////////////////////////////////////////////////////////
// This function is called after every cuda-kernel execution.
//////////////////////////////////////////////////////////////////////
static void onKernelExit(const CUpti_CallbackData *cbInfo) {
	cudaError_t * error = (cudaError_t*) cbInfo->functionReturnValue;
	if ( (*error) != cudaSuccess ) {
		printf("Kernel Exit Error: %d", (*error));
	}

#if INJ_MODE == RF_INJECTIONS 
	if (inj_info.areParamsReady) { // Check if this is the kernel of interest 
		if (inj_info.readyToInject && inj_info.writeBeforeRead) { // error was ready to be injected, but the register was overwritten before being read
			printf("Masked: Write before read\n");
			exit(0); // exit the simulation
		} else if (inj_info.readyToInject && !inj_info.errorInjected) { // error was ready to be injected, but was never injected 
			printf("Masked: Error was never read\n");
			exit(0); // exit the simulation
		}
	}
#endif


#if TIMING
	gettimeofday(&end, NULL);

	long seconds, useconds;    
	seconds  = end.tv_sec  - start.tv_sec;
	useconds = end.tv_usec - start.tv_usec;
	float mTime = ((seconds) * 1000 + useconds/1000.0);
	printf("\nTime for %s:  %f ms\n", cbInfo->symbolName, mTime);
	mTotalTime += mTime;
#endif

}
 ///////////////////////////////////////////////////////////////////////////////////
///
///  Simply lookup the basic block in our dictionary, get its "weight" feild
///  and increment it.
///
///////////////////////////////////////////////////////////////////////////////////
__device__ void sassi_basic_block_entry(SASSIBasicBlockParams *bb)
{
  BLOCK **blockStr = sassi_cfg_blocks->getOrInit((int64_t)bb, [](BLOCK **bpp) { assert(0); });
  atomicAdd(&((*blockStr)->weight), 1);
}

///////////////////////////////////////////////////////////////////////////////////
///
///	Print Kernel Time  
///	Print the graph out in "dot" format.  
///  E.g., use:
///
///       dot -Tps -o graph.ps sassi-cfg.dot 
///
///  to render the graph in postscript.
///
///////////////////////////////////////////////////////////////////////////////////

static void sassi_finalize(sassi::lazy_allocator::device_reset_reason reason)
{
#if TIMING
	printf("\nTotal kernel time: %f ms\n", mTotalTime);
#endif
	cudaDeviceSynchronize();
	FILE *cfgFile = fopen("sassi-cfg.dot", "w");
	sassi_cfg->map([cfgFile](int64_t k, CFG* &cfg) {
	fprintf(cfgFile, "digraph %s {\n", cfg->fnName);
	double weightMax = 0.0;
	for (int bb = 0; bb < cfg->numBlocks; bb++) {
		BLOCK *block = &(cfg->blocks[bb]);
		weightMax = std::max(weightMax, (double)block->weight);
 	}
	for (int bb = 0; bb < cfg->numBlocks; bb++) {
	BLOCK *block = &(cfg->blocks[bb]);
	int per = block->isExit ? 3 : 1;
	int boxWeight = 100 - std::round(100.0 * ((double)block->weight / weightMax));
	int fontWeight = boxWeight > 40 ? 0 : 100;
	fprintf(cfgFile, "\tBB%d [style=filled,fontcolor=gray%d,shape=box,"
		"peripheries=%d,color=gray%lld,label=\"BB%d : %d ins\"];\n", 
		block->id, fontWeight, per, block->weight, block->id, block->numInstrs);
	}
	for (int bb = 0; bb < cfg->numBlocks; bb++) {
		BLOCK *block = &(cfg->blocks[bb]);
		for (int s = 0; s < block->numSuccs; s++) {
		fprintf(cfgFile, "\tBB%d -> BB%d;\n", block->id, block->succs[s]);
		}
	}
	fprintf(cfgFile, "}\n");
	});
	fclose(cfgFile);
}

static sassi::lazy_allocator injectorInit(sassi_init, sassi_finalize, onKernelEntry, onKernelExit); 

