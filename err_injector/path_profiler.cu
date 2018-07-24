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
#include <map>
#include <fstream>
#include <cupti.h>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include "sassi_intrinsics.h"
#include "sassi_lazyallocator.hpp"
#include "sassi_dictionary.hpp"
#include <sassi/sassi-function.hpp>
#include <sassi/sassi-core.hpp>
#include <sassi/sassi-regs.hpp>


// 8Mb of space for CFG information.
#define POOLSIZE (1024 * 1024 * 8)
#define MAX_FN_STR_LEN 64
#define MAX_KNAME_SIZE 200
#define MAX_NUM_BBS	500
#define MAX_PATH_NUMBER 200
// flatten thread id
__device__ int get_flat_tid() {
	int tid_b = threadIdx.x + (blockDim.x * (threadIdx.y + 
  				(threadIdx.z * blockDim.y))); // thread id within a block
	int bid = blockIdx.x + (gridDim.x * (blockIdx.y + 
  				(blockIdx.z * gridDim.y))); // block id 
	int tid = tid_b + (bid * blockDim.x * blockDim.y * blockDim.z);
	return tid;
}

typedef struct {
	char kernelName[MAX_KNAME_SIZE];
	uint64_t bb_increments[500];
} path_sum_info_t;

struct PathDesc {
	uint16_t pathId;
	uint16_t intervalStart;
	uint16_t intervalEnd;
};

static __managed__ unsigned int interval_size;
__managed__ unsigned long long AppDynInstCounter;
__managed__ path_sum_info_t path_sums[50];
__managed__ int current_kname_index;
__managed__ int last_bb_executed[1024*1024];
__managed__ int max_path_id;

__managed__ sassi::dictionary<int64_t, PathDesc> *path_tracker;
__managed__ sassi::dictionary<int64_t, unsigned long long> *path_count;

std::map<std::string, int> knameIndexMap;

void parse_params(std::string filename) {
	std::ifstream interval_filestream("interval.txt", std::ifstream::in);
	if(interval_filestream.is_open())
		interval_filestream >> interval_size;
	interval_filestream.close();
	printf("interval size is %d\n", interval_size);
	std::ifstream ifs (filename.c_str(), std::ifstream::in);
	int num_kernels = 0;
	if (ifs.is_open())
	{
		ifs >> num_kernels;
		printf("num kernels: %d\n", num_kernels);
		assert(num_kernels > 0);
		//path_sums = (path_sum_info_t *)malloc(sizeof(path_sum_info_t) * num_kernels);
		for (int k_idx=0; k_idx<num_kernels; k_idx++)
		{
			int num_bbs = 0;
			ifs >> path_sums[k_idx].kernelName;
			printf("  kName %s\n", path_sums[k_idx].kernelName);
			knameIndexMap[path_sums[k_idx].kernelName] = k_idx;
			ifs >> num_bbs;
			printf("   num bbs: %d\n", num_bbs);
			assert(num_bbs > 0);
			//path_sums[k_idx].bb_increments = (uint64_t *)malloc(sizeof(uint64_t) * num_bbs);
			uint16_t last_bb_from = 65535;
			for (int bb_idx=0; bb_idx<num_bbs; bb_idx++)
			{
				uint16_t bb_from = 65535, bb_to = 65535, inc_val = 65535;
				ifs >> bb_from >> bb_to >> inc_val;
				printf("    from %u to %u, inc %u\n", bb_from, bb_to, inc_val);
			//	bb_increments[x] = [64-48:bb_to_2, 47-32:inc_2, 31-16:bb_to, 15-0:inc]
			//	|63................48|47................32|31................16|15................0|
			//	|--------------------|--------------------|--------------------|-------------------|
			//	|      NEXT BB       |     INC FOR BB     |       NEXT BB      |     INC FOR BB    |
			//	|--------------------|--------------------|--------------------|-------------------|
			//	|---------2ND DIVERGENCE (IF ANY)---------|-----------DEFAULT NEXT PATH------------|
				if(last_bb_from != bb_from){
					path_sums[k_idx].bb_increments[bb_from] = ((uint64_t)(bb_to) << 16) | inc_val;
/*					uint16_t bbnext = (uint16_t) (((1 << 16)-1) & (path_sums[k_idx].bb_increments[bb_from] >> 16));
					uint16_t inc = (uint16_t) (((1 << 16) -1) & (path_sums[k_idx].bb_increments[bb_from]));
					printf(" 1--  from %u to %u, inc %u\n", bb_from, bbnext, inc); */
				}
				else {
					uint64_t tmp = path_sums[k_idx].bb_increments[bb_from];
					path_sums[k_idx].bb_increments[bb_from] = 
							((uint64_t)(bb_to) << 48) | ((uint64_t)inc_val << 32) | tmp;
/*					uint16_t bbnext = (uint16_t) (((1 << 16)-1) & (path_sums[k_idx].bb_increments[bb_from] >> 48));
					uint16_t inc = (uint16_t) (((1 << 16) -1) & (path_sums[k_idx].bb_increments[bb_from] >> 32));
					printf("  2-- from %u to %u, inc %u\n", bb_from, bbnext, inc); */

				}

				last_bb_from = bb_from;
			}
		}
	}
}
		
// Create a memory pool that we can populate on the device and read on the host.
//static __managed__ uint8_t sassi_mempool[POOLSIZE];
//static __managed__ int     sassi_mempool_cur;

std::ofstream bb_ofs;

// A dictionary of paths per thread.
//static __managed__ sassi::dictionary<int64_t, PATH_TRACKER> *sassi_path;

// A dictionary of counts for each path.
//static __managed__ sassi::dictionary<int64_t, PATH_COUNT> *sassi_path_count;

///////////////////////////////////////////////////////////////////////////////////
///
///  Allocate data from the UVM mempool.
///
///////////////////////////////////////////////////////////////////////////////////
/*__device__ void *simple_malloc(size_t sz)
{
  int ptr = atomicAdd(&sassi_mempool_cur, sz);
  assert ((ptr + sz) <= POOLSIZE);
  return (void*) &(sassi_mempool[ptr]);
}
*/
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
/*
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
	blockPtr->InPred = blockParam->liveIn->pred;
	blockPtr->InCC = blockParam->liveIn->cc;
	blockPtr->InUnused = blockParam->liveIn->unused;
	blockPtr->OutPred = blockParam->liveOut->pred;
	blockPtr->OutCC = blockParam->liveOut->cc;
	blockPtr->OutUnused = blockParam->liveOut->unused;

//	blockPtr->liveOut = blockParam->liveOut;
//	simple_strncpy(blockPtr->fnName, blockParam->GetFnName());
      });
  }
}
*/

///////////////////////////////////////////////////////////////////////////////////
///
///  Simply lookup the basic block in our dictionary, get its "weight" feild
///  and increment it.
///
///////////////////////////////////////////////////////////////////////////////////
__device__ void sassi_basic_block_entry(SASSIBasicBlockParams *bb)
{
	if (simple_strncmp(path_sums[current_kname_index].kernelName,bb->GetFnName()) !=0)
		return;
	int threadId = get_flat_tid();
	bool is_backedge = false;
	if ((threadId % 32) == 0) {
		unsigned int interval = AppDynInstCounter/interval_size;
		int64_t path_tracker_idx = threadId/32;
		int bb_id = bb->GetID();
		int last_bb_id = last_bb_executed[path_tracker_idx];

//		if (bb->IsEntryBlock())
//		{
			PathDesc *tracker = path_tracker->getOrInit(path_tracker_idx, [interval](PathDesc *pd) {
			pd->pathId = 0;
			pd->intervalStart = interval;
			pd->intervalEnd = 65535;
			});

		//printf("threadID: %d reporting\n", threadId);
//		}
		//FIXME:: What if somehow this entry already existed?

		// update path value based on inc_val
//		else {
//			PathDesc *tracker = *(path_tracker->getOrInit(path_tracker_idx, [] (PathDesc **pd) {
//				assert(0); //we should have already initialized tracker for this threadIdx
//			}));

			uint64_t bbNext_and_Inc = path_sums[current_kname_index].bb_increments[last_bb_id];
			uint16_t bbNext = (uint16_t) (((1 << 16)-1) & (bbNext_and_Inc >> 16));
			uint16_t inc = 0;
			if (bb_id == bbNext)
				inc = (uint16_t) (((1 << 16) -1) & (bbNext_and_Inc));
			else
			{
				bbNext = (uint16_t) (((1 << 16) -1) & (bbNext_and_Inc >> 48));
				if (bb_id == bbNext) {
					inc = (uint16_t) (((1 << 16) -1) & (bbNext_and_Inc >> 32));
				}
				else // can't find an edge from last bb to this bb, it might be a backedge
				{
					is_backedge = true;
				}
			}
			if (!is_backedge)
				(tracker)->pathId += inc;
			else
			{
				printf("BACK EDGE------------------\n");
				uint16_t path_id = (tracker)->pathId;
				__threadfence();
				if (path_id > max_path_id)
					max_path_id = path_id;
				// PATH KEY:
				//|...16 bits...|...16 bits...|...16 bits...|...16 bits...|
				//|-------------|-------------|-------------|-------------|
				//|   PATH ID   |    16b'0    |intervalStart| intervalEnd |
				//|-------------|-------------|-------------|-------------|
				int64_t path_key = ((int64_t)(path_id) << 48) | ((tracker)->intervalStart<<16) | interval;
				unsigned long long* PathCount = path_count->getOrInit(path_key, [] (unsigned long long *count) {
					*count = 1;
				});
				atomicAdd(PathCount, 1LL);
				(tracker)->pathId = 0;
				(tracker)->intervalStart = interval;
				(tracker)->intervalEnd = 65535;
			}
//		}
		
		last_bb_executed[path_tracker_idx] = bb_id;
		if (bb->IsExitBlock())
		{
			//printf("exit block\n");
//			PathDesc *tracker = *(path_tracker->getOrInit(path_tracker_idx, [] (PathDesc **pd) { }));
			uint16_t path_id = (tracker)->pathId;
			__threadfence();
			if (path_id > max_path_id)
				max_path_id = path_id;
			int64_t path_key = ((int64_t)(path_id) << 48) | ((tracker)->intervalStart<<16) | interval;
			//printf("path_id: %u, intervalStart %u, intervalStop %u\n",
			//	path_id, (tracker)->intervalStart, interval);
			unsigned long long* PathCount = path_count->getOrInit(path_key, [] (unsigned long long *count) {
					*count = 0;
				});
				atomicAdd(PathCount, 1LL);

//			atomicAdd(&(path_count[path_id]), 1LL);
		}
	}
}

__device__ void sassi_before_handler(SASSIBeforeParams* bp)
{
	if (bp->GetInstrWillExecute()) {
		atomicAdd(&AppDynInstCounter, 1LL);
	}
}

// This function will be exected before a kernel is launced
static void onKernelEntry(const CUpti_CallbackData *cbInfo) {
	cudaError_t * error = (cudaError_t*) cbInfo->functionReturnValue;

	if ( (*error) != cudaSuccess ) {
		printf("Kernel Entry Error: %d\n", (*error));
	}
	path_tracker->clear();
	std::string kName = cbInfo->symbolName; // name of kernel
	bb_ofs << "kernel," << kName << "\n";
	current_kname_index = knameIndexMap[kName];
	printf("kernel %s: index: %d\n", kName.c_str(), current_kname_index);

}

// This function will be exected after the kernel exits 
static void onKernelExit(const CUpti_CallbackData *cbInfo) {
	cudaDeviceSynchronize();
	cudaError_t *error = (cudaError_t*) cbInfo->functionReturnValue;
	if ((*error) != cudaSuccess) {
		printf("Kernel Exit error: %d\n", (*error));
	}
	std::string kName = cbInfo->symbolName;
	//printf("printing path profile after %s\n", kName.c_str());
	path_count->map([kName](int64_t k, unsigned long long &c) {
		//printf("PATH  => %lld\n", c);//(uint16_t)(k>>48), *c);
		bb_ofs << "kname," << kName << ",path_id," << (uint16_t) (k>>48) << ",intervalStart,"
			<< (uint16_t)((k>>16)&(0x00ff)) << ",intervalEnd," <<(uint16_t)((k)&(0x000000ff)) <<",count,"
			<< c << "\n";
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
	bb_ofs.close();
}

///////////////////////////////////////////////////////////////////////////////////
///
///  Initialize the UVM memory pool and our two dictionaries.  
///
///////////////////////////////////////////////////////////////////////////////////
static void sassi_init()
{
	parse_params("cfgs.txt");
	bb_ofs.open("path_profile.txt", std::ofstream::out);
//	sassi_mempool_cur = 0;
//	bzero(path_count, sizeof(path_count));
	path_tracker = new sassi::dictionary<int64_t, PathDesc>();
	path_count = new sassi::dictionary<int64_t, unsigned long long>();
	AppDynInstCounter = 0;
	current_kname_index = 0;
	max_path_id = 0;
	bzero(last_bb_executed, sizeof(last_bb_executed));
//	bzero(sassi_mempool, sizeof(sassi_mempool));
}


///////////////////////////////////////////////////////////////////////////////////
///
///  
///
///////////////////////////////////////////////////////////////////////////////////
static sassi::lazy_allocator mapAllocator(sassi_init, sassi_finalize, onKernelEntry, onKernelExit);
