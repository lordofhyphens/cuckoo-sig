#ifndef SIGCACHE_H
#define SIGCACHE_H

#include "sigtable.h"

#include <hemi/hemi.h>
#include <hemi/atomic.h>

// Signature-based cache, meant to work alongside a SigTable.
class SigCache : public CuckooSignatureTable
{
  public:
    HEMI_DEV_CALLABLE_INLINE_MEMBER SigCache(unsigned int * key1, unsigned int * key2, unsigned int * key3,
        unsigned long long int * table, unsigned int slots, unsigned int keySlots,
        unsigned int * hash_a, unsigned int * hash_b,
        unsigned int hash_count, unsigned int * top) :
      CuckooSignatureTable(key1, key2, key3, table, slots, keySlots, NULL, 0, 
          hash_a, hash_b, hash_count, NULL, NULL, NULL) {}

    // This constructor adds-on to a CuckooSignatureTable.
    HEMI_DEV_CALLABLE_INLINE_MEMBER SigCache(CuckooSignatureTable parent, unsigned long long int * table, 
        unsigned int slots) : CuckooSignatureTable(parent, table, slots, NULL, 0) {}


    // Just insert without modifying the keytables. We also don't care if the insert fails (it's a cache)
    HEMI_DEV_CALLABLE_INLINE_MEMBER unsigned int insert(unsigned int key1, unsigned int key2, unsigned int key3, unsigned int v) {
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
        }
        return v; // we don't care about the entries that are bumped from the table.
      }
    }
};
#endif // SIGCACHE_H 
