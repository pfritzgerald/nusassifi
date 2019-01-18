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
#include <fstream>
#include <cupti.h>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include "sassi_intrinsics.h"
#include "sassi_lazyallocator.hpp"
#include "sassi_dictionary.hpp"
#include <sassi/sassi-function.hpp>


// 8Mb of space for CFG information.
#define POOLSIZE (1024 * 1024 * 8)
#define MAX_FN_STR_LEN 64
// flatten thread id
__device__ int get_flat_tid() {
  int tid_b = threadIdx.x + (blockDim.x * (threadIdx.y + (threadIdx.z * blockDim.y))); // thread id within a block
  int bid = blockIdx.x + (gridDim.x * (blockIdx.y + (blockIdx.z * gridDim.y))); // block id 
	int tid = tid_b + (bid * blockDim.x * blockDim.y * blockDim.z);
	return tid;
}
// Create a memory pool that we can populate on the device and read on the host.
static __managed__ uint8_t sassi_mempool[POOLSIZE];
static __managed__ int     sassi_mempool_cur;

std::ofstream bb_ofs;

//A structure to keep track of the path of each thread
struct PATH_TRACKER {
  int8_t path[100];
  int num_bbs_seen;
};

struct PATH_COUNT {
	int count;
	int8_t path[100];
};

// A dictionary of paths per thread.
static __managed__ sassi::dictionary<int64_t, PATH_TRACKER> *sassi_path;

// A dictionary of counts for each path.
static __managed__ sassi::dictionary<int64_t, PATH_COUNT> *sassi_path_count;

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
	int threadId = get_flat_tid();
	if ((threadId % 32) == 0) {
		PATH_TRACKER *path_track = sassi_path->getOrInit((int64_t)threadId, [bb](PATH_TRACKER *pt) {
			 pt->num_bbs_seen = 0;
			 });
	
		int bb_count = (path_track)->num_bbs_seen;
		(path_track)->path[bb_count] =(int8_t)(bb->GetID());
		(path_track)->path[bb_count+1] = -1;
		(path_track)->num_bbs_seen = bb_count +1;

		if (bb->IsExitBlock())
		{

			//(path_track)->path[bb_count+1] = '\0';
			//uint8_t *path = ;
			int64_t key = 0;
			for (int i=0; i<100; i++)
			{
				if (path_track->path[i] == -1) break;
				key = key | (path_track->path[i] << i*8);
			}

	//	simple_strncpy(path, path_track->path);
			PATH_COUNT *path_count = sassi_path_count->getOrInit(key, [path_track](PATH_COUNT *count) {
				count->count = 1;
				for (int i=0; i<100; i++)
				{
					count->path[i] = path_track->path[i];
					if (path_track->path[i] == -1) break;
				}
			});
			atomicAdd(&(path_count->count), 1);
/*			printf("----PATH[0][1][2]: %d-%d-%d--- path_count:%d\n", 
				(path_count)->path[0], (path_count)->path[1],
				(path_count)->path[2], path_count->count);
*/
		}
	}
}

// This function will be exected before a kernel is launced
static void onKernelEntry(const CUpti_CallbackData *cbInfo) {
	sassi_path->clear();
	cudaError_t * error = (cudaError_t*) cbInfo->functionReturnValue;

	if ( (*error) != cudaSuccess ) {
		printf("Kernel Entry Error: %d\n", (*error));
	}

	std::string kName = cbInfo->symbolName; // name of kernel
	bb_ofs << "kernel: " << kName << "\n";

}

// This function will be exected after the kernel exits 
static void onKernelExit(const CUpti_CallbackData *cbInfo) {
//	cudaDeviceSynchronize();
	cudaError_t *error = (cudaError_t*) cbInfo->functionReturnValue;
	if ((*error) != cudaSuccess) {
		printf("Kernel Exit error: %d\n", (*error));
	}

	printf("printing path profile after %s\n", (cbInfo->symbolName));
	sassi_path_count->map([](int64_t &k, PATH_COUNT &v) {
		printf("\n---------\nPATH: ");
		for (int i=0; i<100; i++)
		{
			if (v.path[i] == -1) break;
			printf("-%d", v.path[i]);
		}
		printf(" ====> COUNT: %d\n-----------\n", v.count);
		bb_ofs << (int8_t)v.path[0] << "-" << (int8_t)v.path[1] << " : " << v.count << "\n";
//		fprintf(cfgFile, "count,%d", v);
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
	bb_ofs.open("bb_path_profile.txt", std::ofstream::out);
	sassi_mempool_cur = 0;
	bzero(sassi_mempool, sizeof(sassi_mempool));
	sassi_path_count = new sassi::dictionary<int64_t, PATH_COUNT>();
	sassi_path = new sassi::dictionary<int64_t, PATH_TRACKER>(7919);
}


///////////////////////////////////////////////////////////////////////////////////
///
///  
///
///////////////////////////////////////////////////////////////////////////////////
static sassi::lazy_allocator mapAllocator(sassi_init, sassi_finalize, onKernelEntry, onKernelExit);
