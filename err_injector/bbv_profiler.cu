/*********************************************************************************** \
 * Copyright (c) 2015, NVIDIA open source projects
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * - Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * 
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * 
 * - Neither the name of SASSI nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * This example shows how to use SASSI to inspect the control flow graph.
 *
 * The application code the user instruments should be instrumented with the
 * following SASSI flags: -Xptxas --sassi-function-entry -Xptxas --sassi-bb-entry
 *  
\***********************************************************************************/

#include <algorithm>
#include <assert.h>
#include <cupti.h>
#include <fstream>
#include <map>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <unistd.h>
#include <sassi/sassi-core.hpp>
#include <sassi/sassi-regs.hpp>
#include <sassi/sassi-memory.hpp>
#include "sassi_intrinsics.h"
#include "sassi_lazyallocator.hpp"
#include "sassi_dictionary.hpp"
#include <sassi/sassi-function.hpp>

#include "error_injector.h"

#define MAX_FN_STR_LEN 64
#define MAX_NUM_INTERVALS 600
__managed__ unsigned long long AppDynInstCounter; // count all insts in the app
__managed__ unsigned long long GPRDynInstCounter[MAX_NUM_INTERVALS]; //reset each interval
__managed__ unsigned long long int CycleCounter[MAX_NUM_INTERVALS];
__managed__ long long int start_time, prev_time;

std::map<std::string, int> knameCount;
std::ofstream bb_ofs;
std::string profiler_filename = "basic_block_insts.txt";
std::string interval_filename = "interval.txt";

// Create a memory pool that we can populate on the device and read on the host.
static __managed__ unsigned int interval_size;

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
  char fnName[MAX_FN_STR_LEN];
  int interval;
};

// A dictionary of basic block executions per interval.
static __managed__ sassi::dictionary<int64_t, BLOCK> *sassi_bb_profile;

// Parse interval size from a file. This should be done on host side.
void parse_params(std::string filename) {
 	std::ifstream ifs (filename.c_str(), std::ifstream::in);
	if (ifs.is_open()) 
		ifs >> interval_size; 
	ifs.close();
	printf("----\nFRITZ: interval size is %d\n", interval_size);
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
///////////////////////////////////////////////////////////////////////////////////
///
///  A simple string compre to compare two strings to our UVM malloc'd region.
///
///////////////////////////////////////////////////////////////////////////////////
__device__ int simple_strncmp(char *dest, const char *src)
{
  int i;
  for (i = 0; i < MAX_FN_STR_LEN-1; i++) {
    if (dest[i] != src[i])
	    return -1;
  //  if ((dest[i] == 0) && (src[i] == 0))
//    else if ((dest[i] == 0) || (src[i] == 0))
//	    return -2;
  }
  
	    return 0;
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
//	__device__ void sassi_function_entry(SASSIFunctionParams* fp)
//	{
//	  int numBlocks = fp->GetNumBlocks();
//	  const SASSIBasicBlockParams * const * blocks = fp->GetBlocks();
//	  
//	  CFG *cPtr = *(sassi_cfg->getOrInit((int64_t)fp, [numBlocks, blocks, fp](CFG **cfg) {
//	      CFG *cPtr = (CFG*) simple_malloc(sizeof(CFG));
//	      simple_strncpy(cPtr->fnName, fp->GetFnName());
//	      cPtr->numBlocks = numBlocks;
//	      cPtr->blocks = (BLOCK*) simple_malloc(sizeof(BLOCK) * numBlocks);
//	      *cfg = cPtr;
//	  }));
//	
//	  __threadfence();
//	
//	  for (int bb = 0; bb < numBlocks; bb++) {
//	    const SASSIBasicBlockParams *blockParam = blocks[bb];
//	    BLOCK *blockPtr = &(cPtr->blocks[bb]); 
//	    sassi_cfg_blocks->getOrInit((int64_t)blockParam, [blockParam, blockPtr](BLOCK **bpp) {
//		*bpp = blockPtr;
//		blockPtr->id = blockParam->GetID();
//		blockPtr->weight = 0;
//		blockPtr->isEntry = blockParam->IsEntryBlock();
//		blockPtr->isExit = blockParam->IsExitBlock();
//		blockPtr->numInstrs = blockParam->GetNumInstrs();
//		blockPtr->numSuccs = blockParam->GetNumSuccs();
//		assert(blockParam->GetNumSuccs() <= 2);
//		const SASSIBasicBlockParams * const * succs = blockParam->GetSuccs();
//		for (int s = 0; s < blockParam->GetNumSuccs(); s++) {
//		  blockPtr->succs[s] = succs[s]->GetID();
//		}
//	      });
//	  }
//	}

// This function will be exected before a kernel is launced
static void onKernelEntry(const CUpti_CallbackData *cbInfo) {
	sassi_bb_profile->clear();
	prev_time = (long long int) (clock() - prev_time);
	cudaError_t * error = (cudaError_t*) cbInfo->functionReturnValue;

	if ( (*error) != cudaSuccess ) {
		printf("Kernel Exit Error: %d", (*error));
	}

	// print per thread counters
	std::string kName = cbInfo->symbolName; // name of kernel
	if (knameCount.find(kName) == knameCount.end()) {
		knameCount[kName] = 0;
	} else {
		knameCount[kName] += 1;				
	}

	char numstr[21]; // enough to hold all numbers up to 64-bits
	sprintf(numstr, "%d", knameCount[kName]); // convert int to string
//	printf("kernel %s - invocation %d\n", kName.c_str(), knameCount[kName]);
	//global_ofs << "kernel," << kName << ",invocation," << numstr << "\n";
	bb_ofs << "kernel," << kName << ",invocation," << numstr << ",interval_size," << interval_size << "\n";

}

///////////////////////////////////////////////////////////////////////////////////
///
///  Simply lookup the basic block in our dictionary, get its "weight" feild
///  and increment it.
///
///////////////////////////////////////////////////////////////////////////////////
__device__ void sassi_basic_block_entry(SASSIBasicBlockParams *bb)
{
	unsigned int interval = AppDynInstCounter / interval_size;
//	printf("FRITZ --- BBParams: %lld\n", (int64_t)bb);
	int64_t key = (((int64_t)bb << 32) + interval);
  	BLOCK *blockStr = sassi_bb_profile->getOrInit((int64_t)key, [bb, interval](BLOCK *bpp) { 
		bpp->id = bb->GetID();
		bpp->weight = 0;
		bpp->isEntry = bb->IsEntryBlock();
		bpp->isExit = bb->IsExitBlock();
		bpp->numInstrs = bb->GetNumInstrs();
		bpp->numSuccs = bb->GetNumSuccs();
		bpp->interval = interval;
		const SASSIBasicBlockParams * const * succs = bb->GetSuccs();
		for(int i=0; i < bb->GetNumSuccs(); i++) {
			bpp->succs[i] = succs[i]->GetID();
		}
		assert(bb->GetNumSuccs() <= 2);
		simple_strncpy(bpp->fnName, bb->GetFnName());
//		printf("FRITZ ::: liveOut maxRRegUsed: %d\n", bb->liveOut->maxRRegUsed);
	 });
	assert(blockStr->id == bb->GetID());
	assert(blockStr->numInstrs == bb->GetNumInstrs());
//	assert(simple_strncmp(blockStr->fnName, bb->GetFnName()) == 0);
  atomicAdd(&(blockStr->weight), 1);
}
// This function will be called after every REG/MEM SASS instruction gets executed 
__device__ void sassi_before_handler(SASSIBeforeParams* bp, SASSIMemoryParams *mp, SASSIRegisterParams *rp) {
//	printf("FRITZ::::Will Execute\n");
	//unsigned long long currGPRInstCount =
	if (bp->GetInstrWillExecute()) {
		unsigned long long curDynInstCount = atomicAdd(&AppDynInstCounter, 1LL);
		unsigned int interval = curDynInstCount / interval_size;
		if (curDynInstCount % interval_size == 0) {
			CycleCounter[interval] = (unsigned long long int)(clock64() - start_time - prev_time);
			start_time = clock64();
			prev_time = 0;
		}

		if (has_dest_reg(rp)) {
			atomicAdd(&(GPRDynInstCounter[interval]), 1ULL);
		}
	}
}

// This function will be exected after the kernel exits 
static void onKernelExit(const CUpti_CallbackData *cbInfo) {
	prev_time = (long long int) clock();
	sassi_bb_profile->map([](int64_t &k,
				BLOCK &bb) {
			bb_ofs  << bb.id << "," << bb.fnName << "," << bb.numInstrs 
			<< "," << bb.interval << "," << bb.weight << "," 
			<< bb.numSuccs << ",";
			for (int i=0; i<bb.numSuccs; i++)
			{
				bb_ofs << bb.succs[i] << ",";
			}
			bb_ofs << "\n";
			});
}


///////////////////////////////////////////////////////////////////////////////////
///
///  Print the graph out in "dot" format.  
///  E.g., use:
///
///       dot -Tps -o graph.ps sassi-cfg.dot 
///
///  to render the graph in postscript.
///
///////////////////////////////////////////////////////////////////////////////////
static void sassi_finalize(sassi::lazy_allocator::device_reset_reason unused)
{
	cudaDeviceSynchronize();
	unsigned int max_interval = AppDynInstCounter/interval_size;
	for (unsigned i = 0; i <= max_interval; i++) {
		bb_ofs << "INTERVAL," << i << ",NUMGPRINSTS,"<< GPRDynInstCounter[i]<<",CYCLES," << (unsigned long long
				int)(CycleCounter[i]) << "\n";
    	}

	bb_ofs.close();
}

///////////////////////////////////////////////////////////////////////////////////
///
///  Initialize the UVM memory pool and our two dictionaries.  
///
///////////////////////////////////////////////////////////////////////////////////
static void sassi_init()
{
	parse_params(interval_filename.c_str());
	bb_ofs.open(profiler_filename.c_str(), std::ofstream::out);
	sassi_bb_profile = new sassi::dictionary<int64_t, BLOCK>();
	bzero(GPRDynInstCounter, sizeof(GPRDynInstCounter));
	bzero(CycleCounter, sizeof(CycleCounter));
	AppDynInstCounter = 0;
	start_time = 0;//clock64();
	prev_time = 0;
}


///////////////////////////////////////////////////////////////////////////////////
///
///  
///
///////////////////////////////////////////////////////////////////////////////////
static sassi::lazy_allocator mapAllocator(sassi_init, sassi_finalize, onKernelEntry, onKernelExit);
