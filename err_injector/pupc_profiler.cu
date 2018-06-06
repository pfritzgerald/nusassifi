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

#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <assert.h>
#include <string>
#include <map>
#include <fstream>
#include <sys/time.h>
#include <cupti.h>

#include <sassi/sassi-core.hpp>
#include <sassi/sassi-regs.hpp>
#include <sassi/sassi-memory.hpp>
#include "sassi/sassi-opcodes.h"
#include "sassi_intrinsics.h"
#include "sassi_dictionary.hpp"
#include "sassi_lazyallocator.hpp"

#define MAX_FN_STR_LEN 64
//__managed__ sassi::dictionary<int64_t, uint64_t> *kname_map;
//__managed__ int fname_counter;
struct PUPC
{
	uint64_t pupc;
	int basic_block_id;
	char fnName[MAX_FN_STR_LEN];
	SASSIInstrOpcode opcode;
	int32_t numGPRDsts;
	int32_t numGPRSrcs;
	int GPRDsts[SASSI_NUMGPRDSTS];
	int GPRSrcs[SASSI_NUMGPRSRCS];
	bool isMem;
	unsigned long long weight;
};

std::ofstream pupc_ofs;
__managed__ sassi::dictionary<uint64_t, PUPC> *sassi_pupcs;  

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

// output profile filenames for global and non-global memory accesses
std::string pupcs_filename = "pupcs.txt";

// This function will be called after every SASS instruction gets executed 
__device__ void sassi_after_handler(SASSIAfterParams* ap, SASSIRegisterParams *rp) {
	//	printf("FRITZ::::Will Execute\n");
	uint64_t pupc = ap->GetPUPC();
	PUPC *pc_entry = sassi_pupcs->getOrInit(pupc, [ap, pupc, rp] (PUPC* inst) {
		inst->basic_block_id = ap->GetBBID();
		inst->pupc = pupc;
/*		int current_fn_counter = fname_counter;
		uint64_t kname_code = kname_map->getOrInit((int64_t)(ap->GetFnName()), 
						[current_fn_counter] (int knameCode) {
			knameCode = current_fn_counter;
			current_fn_counter += 1;
		};
		fname_counter = current_fn_counter;*/
 
		simple_strncpy(inst->fnName, ap->GetFnName());
		//inst->fnName = kname_code;
		inst->opcode = ap->GetOpcode();
		inst->isMem = ap->IsMem();
		inst->numGPRDsts = rp->GetNumGPRDsts();
		inst->numGPRSrcs = rp->GetNumGPRSrcs();
		for (int i=0; i<rp->GetNumGPRSrcs(); i++)
			inst->GPRSrcs[i] = rp->GetRegNum(rp->GetGPRSrc(i));
		for (int i=0; i<rp->GetNumGPRDsts(); i++)
			inst->GPRDsts[i] = rp->GetRegNum(rp->GetGPRDst(i));
	});
	atomicAdd(&(pc_entry->weight), 1);
}

static void sassi_init()
{
	pupc_ofs.open(pupcs_filename.c_str(), std::ofstream::out);
	sassi_pupcs = new sassi::dictionary<uint64_t, PUPC>();
	//kname_map = new sassi::dictionary<int64_t, uint64_t>();
	//fname_counter = 0;
}

static void sassi_finalize(sassi::lazy_allocator::device_reset_reason reason)
{
	cudaDeviceSynchronize();

	sassi_pupcs->map([](uint64_t &k, PUPC &inst) {
		pupc_ofs << "PUPC," << std::hex << inst.pupc << ",BBId,"
		<< std::dec << inst.basic_block_id 
		<< ",fnName," << inst.fnName << ",opcode," 
		<< SASSIInstrOpcodeStrings[inst.opcode] << ",isMem," << inst.isMem << ",weight," 
		<< inst.weight << ",numGPRSrcs," << inst.numGPRSrcs <<",GPRSrcs,";
		for (int i=0; i<inst.numGPRSrcs; i++)
			pupc_ofs << std::dec << inst.GPRSrcs[i] << ",";
		pupc_ofs << "numGPRDsts," << inst.numGPRDsts << ",GPRDsts,";
		for (int i=0; i<inst.numGPRDsts; i++)
			pupc_ofs << std::dec << inst.GPRDsts[i] << ",";
		pupc_ofs << "\n";
	});
/*	kname_map->map([](int64_t &k, uint64_t &code) {
		pupc_ofs << "kernel," << (char *)k << ",code," << code << "\n";
	});*/
	pupc_ofs.close();
}

static sassi::lazy_allocator profilerInit(sassi_init, sassi_finalize); 
