
#define DEBUG
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

HEMI_KERNEL(testSigCacheInsert)
  (unsigned int key1, unsigned int key2, unsigned key3, SigCache ca, unsigned int max)
{
  int offset = hemiGetElementOffset();
  int stride = hemiGetElementStride();
  
  for(int opt = offset; opt < max; opt += stride)
  {
    int r = ca.insert(opt, key1, key2);
    printf("%s, %d: For key (%u, %u, %u):%u, -> %u, %u, %u\n", __FILE__, __LINE__, opt,key1, key2, r, ca.keyPt1[r], ca.keyPt2[r], ca.keyPt3[r]);
    printf("z: %u, %u\n", ca.retrieve(opt,key1,key2), r);
  } 
}
// Create some arrays
int main() 
{

  // Array references for GPGPU. To make this easier to use, need to study and
  // borrow from hemi/array.h.

  // scalars
  unsigned int *g_cacheTop, *g_cacheFreeCount;

  // arrays
  unsigned int *g_cache_key1, *g_cache_key2, *g_cache_key3;
  unsigned int *g_hash_a, *g_hash_b, *h_hash_a, *h_hash_b;
  unsigned long long int *g_cache;
  unsigned int *g_cacheFree;

  const unsigned int SLOTS = 500;
  const unsigned int CACHE_SIZE = 150;
  const unsigned int HASH_FUNCS = 5;

  // all the memory needed, allocated.
  //
  srand(time(NULL));

  checkCuda( cudaMalloc((void**)&g_cacheTop, sizeof(unsigned int)) );


  checkCuda( cudaMallocHost((void**)&h_hash_a, sizeof(unsigned int)*HASH_FUNCS) );
  checkCuda( cudaMallocHost((void**)&h_hash_b, sizeof(unsigned int)*HASH_FUNCS) );
  checkCuda( cudaMalloc((void**)&g_hash_a, sizeof(unsigned int)*HASH_FUNCS) );
  checkCuda( cudaMalloc((void**)&g_hash_b, sizeof(unsigned int)*HASH_FUNCS) );



  checkCuda( cudaMalloc((void**)&g_cache_key1, sizeof(unsigned int)*CACHE_SIZE) );
  checkCuda( cudaMalloc((void**)&g_cache_key2, sizeof(unsigned int)*CACHE_SIZE) );
  checkCuda( cudaMalloc((void**)&g_cache_key3, sizeof(unsigned int)*CACHE_SIZE) );

  checkCuda( cudaMalloc((void**)&g_cacheFree, sizeof(unsigned int)*CACHE_SIZE) );
  checkCuda( cudaMalloc((void**)&g_cacheFreeCount, sizeof(unsigned int)) );

  checkCuda( cudaMalloc((void**)&g_cache, sizeof(unsigned long long int)*CACHE_SIZE) );

  srand(time(NULL));
  for (int i = 0; i < HASH_FUNCS; i++)
  {
    h_hash_a[i] = rand();
    h_hash_b[i] = rand();
  }


  checkCuda( cudaMemset(g_cache_key1, 0, sizeof(unsigned int)*CACHE_SIZE) );
  checkCuda( cudaMemset(g_cache_key2, 0, sizeof(unsigned int)*CACHE_SIZE) );
  checkCuda( cudaMemset(g_cache_key3, 0, sizeof(unsigned int)*CACHE_SIZE) );
  
  checkCuda( cudaMemset(g_cacheFree, 0, sizeof(unsigned int)*CACHE_SIZE) );
  checkCuda( cudaMemset(g_cacheFreeCount, 0, sizeof(unsigned int)) );

  checkCuda( cudaMemcpy(g_hash_a, h_hash_a, sizeof(unsigned int)*HASH_FUNCS, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpy(g_hash_b, h_hash_b, sizeof(unsigned int)*HASH_FUNCS, cudaMemcpyHostToDevice) );
  checkCuda( cudaFreeHost(h_hash_a) );
  checkCuda( cudaFreeHost(h_hash_b) );

  checkCuda( cudaMemset(g_cacheTop, 0, sizeof(unsigned int)) );


  int blockDim = 128; // blockDim, gridDim ignored by host code
  int gridDim  = std::min<int>(1024, ((SLOTS*3)/2 + blockDim - 1) / blockDim);


  gridDim  = std::min<int>(1024, (CACHE_SIZE + blockDim - 1) / blockDim);
  HEMI_KERNEL_LAUNCH(InitializeCuckooSignatureTable, gridDim, blockDim, 0, 0, g_cache, CACHE_SIZE);
  checkCuda( cudaDeviceSynchronize() );


  int max = 130;

  SigCache test_sigcache = SigCache(g_cache_key1, g_cache_key2, g_cache_key3, g_cache, CACHE_SIZE, CACHE_SIZE, g_hash_a, g_hash_b, HASH_FUNCS, g_cacheTop,g_cacheFree, g_cacheFreeCount);

  gridDim  = std::min<int>(1024, (max + blockDim - 1) / blockDim);

  HEMI_KERNEL_LAUNCH(testSigCacheInsert, gridDim, blockDim, 0, 0, max, 30, 20, test_sigcache, 20);
  checkCuda( cudaDeviceSynchronize() );

  checkCuda( cudaFree(g_cache) );

  checkCuda( cudaFree(g_cache_key1) );
  checkCuda( cudaFree(g_cache_key2) );
  checkCuda( cudaFree(g_cache_key3) );

  checkCuda( cudaFree(g_hash_a) );
  checkCuda( cudaFree(g_hash_b) );
  checkCuda( cudaFree(g_cacheTop) );
  return 0;
}
