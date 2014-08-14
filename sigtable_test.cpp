#include <hemi/hemi.h>
#include <cuda.h>
#include <algorithm>

#include <ctime>
#include <cstdlib>
#include <cstdio>

#include "sigtable.h"
#include "sigcache.h"
HEMI_KERNEL(InitializeCuckooSignatureTable)
    (unsigned long long int* table, unsigned tableSize)
{
  int offset = hemiGetElementOffset();
  int stride = hemiGetElementStride();
  for(int opt = offset; opt < tableSize; opt += stride)
  {
    table[opt] = HEMI_CONSTANT(EMPTY_SLOT);
  }
}

HEMI_KERNEL(testSigTableInsert)
  (unsigned int key1, unsigned int key2, unsigned key3, SigTable st, unsigned int max)
{
  int offset = hemiGetElementOffset();
  int stride = hemiGetElementStride();
  
  for(int opt = offset; opt < max; opt += stride)
  {
    int z = st.insert(opt, key1, key2);
    printf("For key (%u, %u, %u):%u, -> %u, %u, %u\n", opt,key1, key2, z, st.keyPt1[z], st.keyPt2[z], st.keyPt3[z]);
    assert(st.keyPt1[z] == opt);
    assert(st.keyPt2[z] == key1);
    assert(st.keyPt3[z] == key2);
    assert(z == st.retrieve(opt,key1,key2));
  } 
}

HEMI_KERNEL(testSigCacheInsert)
  (unsigned int key1, unsigned int key2, unsigned key3, SigTable st, SigCache ca, unsigned int max)
{
  int offset = hemiGetElementOffset();
  int stride = hemiGetElementStride();
  
  for(int opt = offset; opt < max; opt += stride)
  {
    int z = st.retrieve(opt, key1, key2);
    ca.insert(opt, key1, key2, z);
    printf("For key (%u, %u, %u):%u, -> %u, %u, %u\n", opt,key1, key2, z, st.keyPt1[z], st.keyPt2[z], st.keyPt3[z]);
    assert(st.keyPt1[z] == opt);
    assert(st.keyPt2[z] == key1);
    assert(st.keyPt3[z] == key2);
    assert(z == ca.retrieve(opt,key1,key2));
  } 
}
// Create some arrays
int main() 
{

  // Array references for GPGPU. To make this easier to use, need to study and
  // borrow from hemi/array.h.

  // scalars
  bool * g_rebuild; 
  unsigned int *g_top, *g_stashTop;

  // arrays
  unsigned int *g_key1, *g_key2, *g_key3;
  unsigned int *g_hash_a, *g_hash_b, *h_hash_a, *h_hash_b;
  unsigned long long int *g_table, *g_cache;
  unsigned long long int *g_stash;

  const unsigned int SLOTS = 500;
  const unsigned int CACHE_SIZE = 150;
  const unsigned int STASH_SIZE = 50;
  const unsigned int HASH_FUNCS = 5;

  // all the memory needed, allocated.
  //
  srand(time(NULL));

  checkCuda( cudaMalloc((void**)&g_rebuild, sizeof(bool)) );
  checkCuda( cudaMalloc((void**)&g_top, sizeof(unsigned int)) );
  checkCuda( cudaMalloc((void**)&g_stashTop, sizeof(unsigned int)) );
  checkCuda( cudaMallocHost((void**)&h_hash_a, sizeof(unsigned int)*HASH_FUNCS) );
  checkCuda( cudaMallocHost((void**)&h_hash_b, sizeof(unsigned int)*HASH_FUNCS) );
  checkCuda( cudaMalloc((void**)&g_hash_a, sizeof(unsigned int)*HASH_FUNCS) );
  checkCuda( cudaMalloc((void**)&g_hash_b, sizeof(unsigned int)*HASH_FUNCS) );

  checkCuda( cudaMalloc((void**)&g_key1, sizeof(unsigned int)*SLOTS) );
  checkCuda( cudaMalloc((void**)&g_key2, sizeof(unsigned int)*SLOTS) );
  checkCuda( cudaMalloc((void**)&g_key3, sizeof(unsigned int)*SLOTS) );

  checkCuda( cudaMalloc((void**)&g_table, sizeof(unsigned long long int)*((SLOTS*3)/2)) );
  checkCuda( cudaMalloc((void**)&g_cache, sizeof(unsigned long long int)*CACHE_SIZE) );
  checkCuda( cudaMalloc((void**)&g_stash, sizeof(unsigned long long int)*STASH_SIZE) );

  srand(time(NULL));
  for (int i = 0; i < HASH_FUNCS; i++)
  {
    h_hash_a[i] = rand();
    h_hash_b[i] = rand();
  }

  checkCuda( cudaMemset(g_key1, 0, sizeof(unsigned int)*SLOTS) );
  checkCuda( cudaMemset(g_key2, 0, sizeof(unsigned int)*SLOTS) );
  checkCuda( cudaMemset(g_key3, 0, sizeof(unsigned int)*SLOTS) );
  checkCuda( cudaMemcpy(g_hash_a, h_hash_a, sizeof(unsigned int)*HASH_FUNCS, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpy(g_hash_b, h_hash_b, sizeof(unsigned int)*HASH_FUNCS, cudaMemcpyHostToDevice) );
  checkCuda( cudaFreeHost(h_hash_a) );
  checkCuda( cudaFreeHost(h_hash_b) );

  checkCuda( cudaMemset(g_top, 0, sizeof(unsigned int)) );
  checkCuda( cudaMemset(g_stashTop, 0, sizeof(unsigned int)) );

  int blockDim = 128; // blockDim, gridDim ignored by host code
  int gridDim  = std::min<int>(1024, ((SLOTS*3)/2 + blockDim - 1) / blockDim);

  // Set up device memory arrays

  HEMI_KERNEL_LAUNCH(InitializeCuckooSignatureTable, gridDim, blockDim, 0, 0, g_table, (SLOTS*3)/2);
  gridDim  = std::min<int>(1024, (CACHE_SIZE + blockDim - 1) / blockDim);
  HEMI_KERNEL_LAUNCH(InitializeCuckooSignatureTable, gridDim, blockDim, 0, 0, g_cache, CACHE_SIZE);
  gridDim  = std::min<int>(1024, (STASH_SIZE + blockDim - 1) / blockDim);
  HEMI_KERNEL_LAUNCH(InitializeCuckooSignatureTable, gridDim, blockDim, 0, 0, g_stash, STASH_SIZE);
  checkCuda( cudaDeviceSynchronize() );


  int max = 130;

  SigTable test_sigtable = SigTable(g_key1, g_key2, g_key3, g_table, (SLOTS*3)/2, SLOTS, g_stash, 
                                    STASH_SIZE, g_hash_a, g_hash_b, HASH_FUNCS, g_top, 
                                    g_stashTop, g_rebuild);

  SigCache test_sigcache = SigCache(test_sigtable, g_cache, CACHE_SIZE);

  gridDim  = std::min<int>(1024, (max + blockDim - 1) / blockDim);
  HEMI_KERNEL_LAUNCH(testSigTableInsert, gridDim, blockDim, 0, 0, max, 30, 20, test_sigtable, 20);

  HEMI_KERNEL_LAUNCH(testSigCacheInsert, gridDim, blockDim, 0, 0, max, 30, 20, test_sigtable, test_sigcache, 20);


  checkCuda( cudaFree(g_table) );
  checkCuda( cudaFree(g_stash) );
  checkCuda( cudaFree(g_cache) );

  checkCuda( cudaFree(g_key1) );
  checkCuda( cudaFree(g_key2) );
  checkCuda( cudaFree(g_key3) );

  checkCuda( cudaFree(g_hash_a) );
  checkCuda( cudaFree(g_hash_b) );
  checkCuda( cudaFree(g_top) );
  checkCuda( cudaFree(g_stashTop) );
  checkCuda( cudaFree(g_rebuild) );


  return 0;
}
