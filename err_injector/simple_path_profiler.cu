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
#define POOLSIZE (1024 * 1024 * 1024)
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
	uint32_t pathId;
	uint16_t BBStart;
	uint16_t BBEnd;
};

__managed__ path_sum_info_t path_sums[50];
__managed__ int current_kname_index;
__managed__ uint32_t last_bb_executed[1024*1024]; //[..15b'0..1b'ExitBB?..|..16b'BBID..]
__managed__ int max_path_id;

__managed__ sassi::dictionary<int64_t, PathDesc> *path_tracker;
__managed__ sassi::dictionary<int64_t, unsigned long long> *path_count;

__managed__ sassi::dictionary<int64_t, unsigned long long> *pc_path_count;
__managed__ sassi::dictionary<int64_t, unsigned long long> *pupc_count;

std::map<std::string, int> knameIndexMap;

void parse_params(std::string filename) {
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
				}
				else {
					uint64_t tmp = path_sums[k_idx].bb_increments[bb_from];
					path_sums[k_idx].bb_increments[bb_from] = 
							((uint64_t)(bb_to) << 48) | ((uint64_t)inc_val << 32) | tmp;

				}

				last_bb_from = bb_from;
			}
		}
	}
}
		
// Create a memory pool that we can populate on the device and read on the host.
static __managed__ uint8_t sassi_mempool[POOLSIZE];
static __managed__ int     sassi_mempool_cur;

std::ofstream bb_ofs;

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
__device__ int simple_strncmp(char *dest, const char *src)
{
  int i;
  for (i = 0; i < MAX_FN_STR_LEN-1; i++) {
    if (dest[i] != src[i])
	    return -1;
  }
  
	    return 0;
}
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
		int64_t path_tracker_idx = threadId/32;
		int bb_id = bb->GetID();
		uint16_t last_bb_id = (uint16_t)(last_bb_executed[path_tracker_idx] & 0xffff);

		PathDesc *tracker = path_tracker->getOrInit(path_tracker_idx, [bb_id](PathDesc *pd) {
		pd->pathId = 0;
		pd->BBStart = (uint16_t)bb_id;
		pd->BBEnd = 65535;
		});

	//printf("threadID: %d reporting\n", threadId);
	//FIXME:: What if somehow this entry already existed?
		// Checking for self loop
		if ((last_bb_id == (uint16_t)bb_id) && (!bb->IsEntryBlock())) // because last_bb is set to 0 at the kernel exit
		{
			// we have a self loop
			int64_t path_key = ((int64_t)(0) << 32) | ((uint16_t)(bb_id) << 16) | (uint16_t)bb_id;
			unsigned long long* PathCount = path_count->getOrInit(path_key, [] (unsigned long long *count) {
				*count = 0;
			});
			atomicAdd(PathCount, 1);
		}
		// We have not exited the kernel, check if last BB executed was an exit BB
		if ((last_bb_executed[path_tracker_idx] >> 16) == 0x1)
		{
			uint16_t path_id = (tracker)->pathId;
			__threadfence();
			int64_t path_key = ((int64_t)(path_id) << 32) | ((tracker)->BBStart<<16) | (uint16_t)last_bb_id;
			unsigned long long* PathCount = path_count->getOrInit(path_key, [] (unsigned long long *count) {
				assert(0); // it should have existed already
			});
			atomicAdd(PathCount, -1);
		}

		// update path value based on inc_val
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
			else if (!bb->IsEntryBlock())// can't find an edge from last bb to this bb, it must be a backedge
			{
				is_backedge = true;
			}
		}
		if (!is_backedge)
			(tracker)->pathId += inc;
		else
		{
//			printf("--------------BACK EDGE: BB%u->BB%u ------------------\n", last_bb_id, bb_id);
			uint32_t path_id = (tracker)->pathId;
			__threadfence();
			if (path_id > max_path_id)
				max_path_id = path_id;
			// PATH KEY:
			//|..........32 bits..........|...16 bits...|...16 bits...|
			//|---------------------------|-------------|-------------|
			//|          PATH ID          |   BBStart   |    BBEnd    |
			//|---------------------------|-------------|-------------|
			int64_t path_key = ((int64_t)(path_id) << 32) | ((tracker)->BBStart<<16) | (uint16_t)last_bb_id;
			unsigned long long* PathCount = path_count->getOrInit(path_key, [] (unsigned long long *count) {
				*count = 0;
			});
			atomicAdd(PathCount, 1LL);
			(tracker)->pathId = 0;
			(tracker)->BBStart = (uint16_t)bb_id;
			(tracker)->BBEnd = 65535;
		}
	
		last_bb_executed[path_tracker_idx] = 0|(uint16_t)bb_id;
		if (bb->IsExitBlock())
		{
			// A kernel may have multiple exits, therefore the kernel may not exit at this basic block.
			// This is why we keep track of the BB that was marked as an exit
			last_bb_executed[path_tracker_idx] = (0x1<<16) | (uint16_t)(bb_id);
			//printf("exit block\n");
			uint16_t path_id = (tracker)->pathId;
			__threadfence();
			if (path_id > max_path_id)
				max_path_id = path_id;
			int64_t path_key = ((int64_t)(path_id) << 32) | ((tracker)->BBStart<<16) | (uint16_t)bb_id;
			unsigned long long* PathCount = path_count->getOrInit(path_key, [] (unsigned long long *count) {
					*count = 0;
				});
				atomicAdd(PathCount, 1LL);
		}
	}
}

// This function will be called after every SASS instruction gets executed 
__device__ void sassi_after_handler(SASSIAfterParams* ap, SASSIRegisterParams *rp) {
	//	printf("FRITZ::::Will Execute\n");
	if (simple_strncmp(path_sums[current_kname_index].kernelName,ap->GetFnName()) !=0)
		return;
	int threadId = get_flat_tid();
	if ((threadId % 32) != 0)
		return;
	int64_t path_tracker_idx = threadId/32;

	PathDesc *tracker = path_tracker->getOrInit(path_tracker_idx, [](PathDesc *pd) {
		// we should never be here
			printf("WE HAVE SKIPPED path profiling FOR AN ELIGIBLE THREAD\n");
	});

	uint64_t pupc = ap->GetPUPC();
	int64_t pc_count_key = (pupc << 32) | (tracker)->pathId << 16 | (tracker)->BBStart;
	unsigned long long *pc_entry = pc_path_count->getOrInit(pc_count_key, [] (unsigned long long *count) {
		*count = 0;
		});
	atomicAdd((pc_entry), 1LL);

	unsigned long long *pupc_entry = pupc_count->getOrInit(pupc, [] (unsigned long long *count) {
			*count = 0;
			});
	atomicAdd((pupc_entry), 1LL);
}



// This function will be exected before a kernel is launced
static void onKernelEntry(const CUpti_CallbackData *cbInfo) {
	cudaError_t * error = (cudaError_t*) cbInfo->functionReturnValue;

	if ( (*error) != cudaSuccess ) {
		printf("Kernel Entry Error: %d\n", (*error));
	}
	path_tracker->clear();
	cudaDeviceSynchronize();
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
		bb_ofs << "kname," << kName << ",path_id," << (uint32_t) (k>>32) << ",BBStart,"
			<< (uint16_t)((k>>16)&(0x00ff)) << ",BBEnd," <<(uint16_t)((k)&(0x000000ff)) <<",count,"
			<< c << "\n";
	});
	
	pc_path_count->map([kName](int64_t k, unsigned long long &c) {
			bb_ofs << "kNAME," << kName <<",PC,0x" << std::hex <<
			(uint32_t)(k>>32) << ",BBSTART,"
			<< std::dec << (uint16_t)((k>>16)&(0x00ff)) << ",PATH_ID," << 
			(uint16_t)((k)&(0xff))
			<< ",COUNT," << c << "\n";
			});
	pupc_count->map([kName](int64_t k, unsigned long long &c) {
			bb_ofs << "KNAME," << kName << ",PUPC,0x" << std::hex << 
			(uint64_t)(k) << ",COUNT," << std::dec << c << "\n";
			});
	bzero(last_bb_executed, sizeof(last_bb_executed));
	path_count->clear();
	pc_path_count->clear();
	pupc_count->clear();
	sassi_mempool_cur = 0;
	bzero(sassi_mempool, sizeof(sassi_mempool));
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
//	bzero(path_count, sizeof(path_count));
	path_tracker = new sassi::dictionary<int64_t, PathDesc>();
	path_count = new sassi::dictionary<int64_t, unsigned long long>();
	pc_path_count = new sassi::dictionary<int64_t, unsigned long long>();
	pupc_count = new sassi::dictionary<int64_t, unsigned long long>();
//	AppDynInstCounter = 0;
	current_kname_index = 0;
	max_path_id = 0;
	bzero(last_bb_executed, sizeof(last_bb_executed));
	sassi_mempool_cur = 0;
	bzero(sassi_mempool, sizeof(sassi_mempool));
}


///////////////////////////////////////////////////////////////////////////////////
///
///  
///
///////////////////////////////////////////////////////////////////////////////////
static sassi::lazy_allocator mapAllocator(sassi_init, sassi_finalize, onKernelEntry, onKernelExit);
