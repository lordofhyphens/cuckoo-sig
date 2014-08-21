#ifndef SIGCACHE_H
#define SIGCACHE_H

#include "sigtable.h"

#include <hemi/hemi.h>
#include <hemi/atomic.h>

// Signature-based cache, meant to work alongside a SigTable.
class SigCache : public CuckooSignatureTable
{
  unsigned int* freeList; unsigned int* freeCount;
  public:
    HEMI_DEV_CALLABLE_INLINE_MEMBER SigCache(unsigned int * key1, unsigned int * key2, unsigned int * key3,
        unsigned long long int * table, unsigned int slots, unsigned int keySlots,
        unsigned int * hash_a, unsigned int * hash_b,
        unsigned int hash_count, unsigned int * top, unsigned int * free, unsigned int* freeCount) :
      CuckooSignatureTable(key1, key2, key3, table, slots, keySlots, NULL, 0, 
          hash_a, hash_b, hash_count, top, NULL, NULL), freeList(free), freeCount(freeCount) {}

    // This constructor adds-on to a CuckooSignatureTable.
    HEMI_DEV_CALLABLE_INLINE_MEMBER SigCache(CuckooSignatureTable parent, unsigned long long int * table, 
        unsigned int slots) : CuckooSignatureTable(parent, table, slots, NULL, 0) {}

    // Add this slot in the keyarray to the free list.
    HEMI_DEV_CALLABLE_INLINE_MEMBER void pushToFreeList(unsigned int slot)
    {
      unsigned int spot = 0;
      while (hemi::atomicCAS(freeList + spot,  Size(), slot) == Size()) {
        spot++;
      }
      hemi::atomicInc(freeCount, Size());
    }

    
    // Get a value from the freelist
    HEMI_DEV_CALLABLE_INLINE_MEMBER unsigned int popFromFreeList()
    {
      unsigned int spot = 0;
      unsigned int result = 0;
      if (*freeCount == 0)
        return Size();
      while ((result = hemi::atomicExch(spot + freeList, Size())) != Size() && *freeCount > 0)
        spot++;
      hemi::atomicDec(freeCount, 0);
      return result;
    }
    HEMI_DEV_CALLABLE_INLINE_MEMBER unsigned int insert(unsigned int key1, unsigned int key2, unsigned int key3) 
    {
      // Consult the free list.
      unsigned slot = 0; 
      if ((slot = popFromFreeList()) == Size())
      {
        // if free list is empty, increment the main head node.
        slot = hemi::atomicAdd(top, 1);
      }
      assert(slot < keySize); // make sure we haven't overfilled our key arrays.
      unsigned int sig = computeSignature(key1, key2, key3);
      unsigned int result = keyInsert(sig, slot);
      if (result != slot) 
      {
        // we have an eviction, add it to the free list
        pushToFreeList(result);
      }
      keyPt1[slot] = key1;
      keyPt2[slot] = key2;
      keyPt3[slot] = key3;
      return slot; // don't care 
    }
    // Just insert without modifying the keytables. We also don't care if the insert fails (it's a cache)
    HEMI_DEV_CALLABLE_INLINE_MEMBER unsigned int insert(unsigned int key1, unsigned int key2, unsigned int key3, unsigned int v) 
    {
      unsigned int key = computeSignature(key1,key2,key3);
      unsigned int entry = retrieve(key1, key2, key3);
      if (entry < keySize) return entry; // already in the table
      unsigned int oldval = keyInsert(key, v);
      if (oldval != v && oldval < Size()) 
      {
        // clear old key out
        keyPt1[oldval] = HEMI_CONSTANT(EMPTY_KEY);
        keyPt2[oldval] = HEMI_CONSTANT(EMPTY_KEY);
        keyPt3[oldval] = HEMI_CONSTANT(EMPTY_KEY);
        
        // Add this location to the free list.
      }
      return oldval; // return oldval so we can do some maintenance.
    }
};
#endif // SIGCACHE_H ;

