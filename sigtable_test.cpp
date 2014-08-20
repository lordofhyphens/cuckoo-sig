
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

HEMI_KERNEL(testSigTableInsert)
  (unsigned int key1, unsigned int key2, unsigned key3, SigTable st, unsigned int max)
{
  int offset = hemiGetElementOffset();
  int stride = hemiGetElementStride();
  
  for(int opt = offset; opt < max; opt += stride)
  {
    unsigned int z = st.insert(opt, key1, key2);
    printf("%s, %s: For key (%u, %u, %u):%u, -> %u, %u, %u\n", __FILE__, __LINE__, opt,key1, key2, z, st.keyPt1[z], st.keyPt2[z], st.keyPt3[z]);
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
    int r = ca.insert(opt, key1, key2);
    printf("%s, %s: For key (%u, %u, %u):%u, -> %u, %u, %u\n", __FILE__, __LINE__, opt,key1, key2, r, ca.keyPt1[r], ca.keyPt2[r], ca.keyPt3[r]);
    printf("z: %u, %u\n", 0, r);
  } 
}
// Create some arrays
int main() 
{

  // Array references for GPGPU. To make this easier to use, need to study and
  // borrow from hemi/array.h.

  // scalars
  bool * g_rebuild; 
  unsigned int *g_top, *g_stashTop, *g_cacheTop;

  // arrays
  unsigned int *g_key1, *g_key2, *g_key3;
  unsigned int *g_cache_key1, *g_cache_key2, *g_cache_key3;
  unsigned int *g_hash_a, *g_hash_b, *h_hash_a, *h_hash_b;
  unsigned long long int *g_table, *g_cache;
  unsigned int *g_cacheFree;

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

  checkCuda( cudaMalloc((void**)&g_cacheTop, sizeof(unsigned int)) );

  fprintf(stderr, "%s, %d: Allocated memory\n", __FILE__,__LINE__); checkCudaErrors();

  checkCuda( cudaMallocHost((void**)&h_hash_a, sizeof(unsigned int)*HASH_FUNCS) );
  checkCuda( cudaMallocHost((void**)&h_hash_b, sizeof(unsigned int)*HASH_FUNCS) );
  checkCuda( cudaMalloc((void**)&g_hash_a, sizeof(unsigned int)*HASH_FUNCS) );
  checkCuda( cudaMalloc((void**)&g_hash_b, sizeof(unsigned int)*HASH_FUNCS) );

  fprintf(stderr, "%s, %d: Allocated memory\n", __FILE__,__LINE__); checkCudaErrors();

  checkCuda( cudaMalloc((void**)&g_key1, sizeof(unsigned int)*SLOTS) );
  checkCuda( cudaMalloc((void**)&g_key2, sizeof(unsigned int)*SLOTS) );
  checkCuda( cudaMalloc((void**)&g_key3, sizeof(unsigned int)*SLOTS) );

  fprintf(stderr, "%s, %d: Allocated memory\n", __FILE__,__LINE__); checkCudaErrors();

  checkCuda( cudaMalloc((void**)&g_cache_key1, sizeof(unsigned int)*CACHE_SIZE) );
  checkCuda( cudaMalloc((void**)&g_cache_key2, sizeof(unsigned int)*CACHE_SIZE) );
  checkCuda( cudaMalloc((void**)&g_cache_key3, sizeof(unsigned int)*CACHE_SIZE) );

  checkCuda( cudaMalloc((void**)&g_cacheFree, sizeof(unsigned int)*CACHE_SIZE) );


  fprintf(stderr, "%s, %d: Allocated memory\n", __FILE__,__LINE__); checkCudaErrors();

  checkCuda( cudaMalloc((void**)&g_table, sizeof(unsigned long long int)*((SLOTS*3)/2)) );
  checkCuda( cudaMalloc((void**)&g_cache, sizeof(unsigned long long int)*CACHE_SIZE) );
  checkCuda( cudaMalloc((void**)&g_stash, sizeof(unsigned long long int)*STASH_SIZE) );


  fprintf(stderr, "%s, %d: Allocated memory\n", __FILE__,__LINE__); checkCudaErrors();

  srand(time(NULL));
  for (int i = 0; i < HASH_FUNCS; i++)
  {
    h_hash_a[i] = rand();
    h_hash_b[i] = rand();
  }

  fprintf(stderr, "%s, %d: Allocated memory\n", __FILE__,__LINE__); checkCudaErrors();

  checkCuda( cudaMemset(g_key1, 0, sizeof(unsigned int)*SLOTS) );
  checkCuda( cudaMemset(g_key2, 0, sizeof(unsigned int)*SLOTS) );
  checkCuda( cudaMemset(g_key3, 0, sizeof(unsigned int)*SLOTS) );

  fprintf(stderr, "%s, %d: Allocated memory\n", __FILE__,__LINE__); checkCudaErrors();
  checkCuda( cudaMemset(g_cache_key1, 0, sizeof(unsigned int)*CACHE_SIZE) );
  checkCuda( cudaMemset(g_cache_key2, 0, sizeof(unsigned int)*CACHE_SIZE) );
  checkCuda( cudaMemset(g_cache_key3, 0, sizeof(unsigned int)*CACHE_SIZE) );
  
  fprintf(stderr, "%s, %d: Allocated memory\n", __FILE__,__LINE__); checkCudaErrors();
  checkCuda( cudaMemset(g_cacheFree, 0, sizeof(unsigned int)*CACHE_SIZE) );

  checkCuda( cudaMemcpy(g_hash_a, h_hash_a, sizeof(unsigned int)*HASH_FUNCS, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpy(g_hash_b, h_hash_b, sizeof(unsigned int)*HASH_FUNCS, cudaMemcpyHostToDevice) );
  checkCuda( cudaFreeHost(h_hash_a) );
  checkCuda( cudaFreeHost(h_hash_b) );

  fprintf(stderr, "%s, %d: Allocated memory\n", __FILE__,__LINE__); checkCudaErrors();
  checkCuda( cudaMemset(g_top, 0, sizeof(unsigned int)) );
  checkCuda( cudaMemset(g_stashTop, 0, sizeof(unsigned int)) );


  int blockDim = 128; // blockDim, gridDim ignored by host code
  int gridDim  = std::min<int>(1024, ((SLOTS*3)/2 + blockDim - 1) / blockDim);

  fprintf(stderr, "%s, %d: Allocated memory\n", __FILE__,__LINE__); checkCudaErrors();


  // Set up device memory arrays
  HEMI_KERNEL_LAUNCH(InitializeCuckooSignatureTable, gridDim, blockDim, 0, 0, g_table, (SLOTS*3)/2);
  gridDim  = std::min<int>(1024, (CACHE_SIZE + blockDim - 1) / blockDim);
  HEMI_KERNEL_LAUNCH(InitializeCuckooSignatureTable, gridDim, blockDim, 0, 0, g_cache, CACHE_SIZE);
  gridDim  = std::min<int>(1024, (STASH_SIZE + blockDim - 1) / blockDim);
  HEMI_KERNEL_LAUNCH(InitializeCuckooSignatureTable, gridDim, blockDim, 0, 0, g_stash, STASH_SIZE);
  checkCuda( cudaDeviceSynchronize() );

  fprintf(stderr, "%s, %d: Allocated memory\n", __FILE__,__LINE__); checkCudaErrors();

  int max = 130;

  SigTable test_sigtable = SigTable(g_key1, g_key2, g_key3, g_table, (SLOTS*3)/2, SLOTS, g_stash, 
                                    STASH_SIZE, g_hash_a, g_hash_b, HASH_FUNCS, g_top, 
                                    g_stashTop, g_rebuild);

  fprintf(stderr, "%s, %d: Allocated memory\n", __FILE__,__LINE__); checkCudaErrors();
  SigCache test_sigcache = SigCache(g_cache_key1, g_cache_key2, g_cache_key3, g_cache, CACHE_SIZE, CACHE_SIZE, g_hash_a, g_hash_b, HASH_FUNCS, g_cacheTop,g_cacheFree);
  fprintf(stderr, "%s, %d: Allocated memory\n", __FILE__,__LINE__); checkCudaErrors();
  gridDim  = std::min<int>(1024, (max + blockDim - 1) / blockDim);
  HEMI_KERNEL_LAUNCH(testSigTableInsert, gridDim, blockDim, 0, 0, max, 30, 20, test_sigtable, 20);
  checkCuda( cudaDeviceSynchronize() );
  fprintf(stderr, "%s, %d: Allocated memory\n", __FILE__,__LINE__); checkCudaErrors();

  HEMI_KERNEL_LAUNCH(testSigCacheInsert, gridDim, blockDim, 0, 0, max, 30, 20, test_sigtable, test_sigcache, 20);
  checkCuda( cudaDeviceSynchronize() );
  fprintf(stderr, "%s, %d: Allocated memory\n", __FILE__,__LINE__); checkCudaErrors();


  checkCuda( cudaFree(g_table) );
  checkCuda( cudaFree(g_stash) );
  checkCuda( cudaFree(g_cache) );
  
  fprintf(stderr, "%s, %d: Allocated memory\n", __FILE__,__LINE__); checkCudaErrors();

  checkCuda( cudaFree(g_key1) );
  checkCuda( cudaFree(g_key2) );
  checkCuda( cudaFree(g_key3) );
  
  fprintf(stderr, "%s, %d: Allocated memory\n", __FILE__,__LINE__); checkCudaErrors();
  checkCuda( cudaFree(g_cache_key1) );
  checkCuda( cudaFree(g_cache_key2) );
  checkCuda( cudaFree(g_cache_key3) );

  fprintf(stderr, "%s, %d: Allocated memory\n", __FILE__,__LINE__); checkCudaErrors();
  checkCuda( cudaFree(g_hash_a) );
  checkCuda( cudaFree(g_hash_b) );
  checkCuda( cudaFree(g_top) );
  checkCuda( cudaFree(g_stashTop) );
  checkCuda( cudaFree(g_rebuild) );
  fprintf(stderr, "%s, %d: Allocated memory\n", __FILE__,__LINE__); checkCudaErrors();
  return 0;
}
