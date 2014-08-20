#ifndef SIGTABLE_H
#define SIGTABLE_H

#include <hemi/hemi.h>
#include <hemi/atomic.h>

// Straightforward implementation of a cuckoo hashtable under the maximum
// keysize restriction of an unsigned long long int.  In this system, the value is
// a reference to some external array, and the key is a simple hash of the key
// parts. We use 3 keys because that's what the client application wants.
// Generalization for any number of key arguments is left  as an exercise for
// other users. :D
//
// Pattern here is a singleton pattern. Each table should have its own key
// arrays, but could share hash functions.
//
// For decent insert performance, make sure that the key arrays are 2/3rds the
// size of the actual hashtable. The stash shouldn't be any bigger than 100
// entries, as a serial loop is used to check the hash (imposing a penalty
// for not-found responses).
//
// Written by Joseph Lenox
// 
// For CUDA, the user is expected to ensure that all pointers are legal.

// To investigate later: Could filling the stash, explicitly, with table
// entries, then re-adding every item in the table, followed by attempting to
// empty the stash, succeed in opening more space with the current hash
// functions?

// Used in the main cuckoo hash function. It needs to be BIG and prime.
//
#define get_key(entry) ((unsigned int)((entry) >> 32))
#define get_value(entry) ((unsigned int)((entry) & 0xffffffff))

HEMI_DEV_CALLABLE_INLINE unsigned long long make_entry(unsigned int key, unsigned int value) { return ((((unsigned long long)key) << 32) + (value)); }
HEMI_DEFINE_STATIC_CONSTANT(long long int lg_prime,334214459);
HEMI_DEFINE_STATIC_CONSTANT(unsigned int DEFAULT_MAX_ATTEMPTS,120);
HEMI_DEFINE_CONSTANT(unsigned int EMPTY_KEY, 0xffffffff);
struct TableSlot
{
  unsigned int sig;
  unsigned int value;
  HEMI_DEV_CALLABLE_INLINE_MEMBER TableSlot(unsigned int s, unsigned int v) : sig(s), value(v) { }
  HEMI_DEV_CALLABLE_INLINE_MEMBER TableSlot(unsigned long long int t) : sig(get_key(t)), value(get_value(t)) { }
  HEMI_DEV_CALLABLE_INLINE_MEMBER TableSlot& operator=(const unsigned long long int& t) { sig = get_key(t); value = get_value(t); return *this;}
  HEMI_DEV_CALLABLE_INLINE_MEMBER operator unsigned long long int() { return make_entry(sig, value); }

};

HEMI_DEV_CALLABLE_INLINE bool operator==(const TableSlot& lhs, const TableSlot& rhs) 
{ 
  return lhs.sig == rhs.sig && lhs.value == rhs.value;
}
HEMI_DEV_CALLABLE_INLINE bool operator==(const TableSlot& lhs, const unsigned long long int& rhs) 
{ 
  return lhs.sig == get_key(rhs) && lhs.value == get_value(rhs);
}
HEMI_DEV_CALLABLE_INLINE bool operator==(const unsigned long long int& rhs, const TableSlot& lhs) 
{ 
  return lhs.sig == get_key(rhs) && lhs.value == get_value(rhs);
}
// may not actually work as expected.
HEMI_DEFINE_CONSTANT(unsigned long long int EMPTY_SLOT, 0xffffffff00000000);

class CuckooSignatureTable
{
  protected:
    // This needs to be initialized to 0 at the start of the program, and
    // increments with every successful insert. It needs to be in global memory
    // (instead of on the stack) to maintain state between kernel calls.

    unsigned int * top;
    unsigned int * stashTop; // the next free stash slot.
    bool * rebuild; // singleton
    unsigned long long int * table;
    unsigned long long int * stash;
    // Perform the insert on the table, going to the stash if there's no room.
    // If there *still* isn't any room, return the last value we got so the
    // calling function can initiate a rebuild.
    HEMI_DEV_CALLABLE_INLINE_MEMBER unsigned int keyInsert(unsigned int k, unsigned int v)
    {
      TableSlot entry = TableSlot(k,v);

      unsigned int location = hashfunc(k, 0);

      for (unsigned int its = 0; its < MAX_ATTEMPTS; its++)
      {
        // insert new item and check for eviction
        // on gpu we use atomicexch, serial cpu just evicts and uses
        // a temp variable. MP cpu needs to have this part in a critical section.
        //
        //    printf("Trying to put (%x, %x) in location %u\n", get_key(entry), get_value(entry), location);
        entry = TableSlot(hemi::atomicExch(table+location,(unsigned long long int)entry));

        if (entry == HEMI_CONSTANT(EMPTY_SLOT)) return v;

        unsigned int location_0 = hashfunc(entry.sig,0);
        unsigned int location_1 = hashfunc(entry.sig,1);
        unsigned int location_2 = hashfunc(entry.sig,2);
        unsigned int location_3 = hashfunc(entry.sig,3);
        unsigned int location_4 = hashfunc(entry.sig,4);

        //round-robin shift of use of hashfunction.
        if (location == location_0) location = location_1;
        else if (location == location_1) location = location_2;
        else if (location == location_2) location = location_3;
        else if (location == location_3) location = location_4;
        else location = location_0;
      }
      // At this point, we have failed an insertion. Put it in the stash if possible.
      if (insertStash(entry))
        return v;
      // If if it didn't fit in the stash, return the position in the keyarray so the calling function can figure out what to do.
      return entry.value;
    }

    HEMI_DEV_CALLABLE_INLINE_MEMBER bool insertStash(TableSlot entry)
    {
      // need the local variable here to avoid a possible race condition.
      // atomicAdd will reserve a unique #, which we can use to index the stash.
      unsigned int stashNext = hemi::atomicAdd(stashTop, 1);
      if (stashNext < stashSize) {
        stash[stashNext] = entry;
        return true;
      }
      return false;
    }
    // Linear probe of the stash. Returns EMPTY_SLOT on no hit.
    HEMI_DEV_CALLABLE_INLINE_MEMBER TableSlot retrieveStash(unsigned int k1, unsigned int k2, unsigned int k3) const
    {
      for (unsigned int i = 0; i < stashSize; i++) {
        if (keymatch(k1,k2,k3,stash[i]))
          return stash[i];
      }
      return HEMI_CONSTANT(EMPTY_SLOT);
    }

    // Helper function to determine whether or not a TableSlot entry matches a triplet of keys.
    HEMI_DEV_CALLABLE_INLINE_MEMBER bool keymatch(unsigned int key1, unsigned int key2, unsigned int key3, TableSlot entry) const
    {
      return (computeSignature(key1, key2, key3) == entry.sig) 
             && (keyPt1[entry.value] == key1) 
             && (keyPt2[entry.value] == key2) 
             && (keyPt3[entry.value] == key3);
    }

  public:
    unsigned int * keyPt1;
    unsigned int * keyPt2;
    unsigned int * keyPt3;

    // Stores our hash functions.
    unsigned int * hash_a;
    unsigned int * hash_b;

    // size of our key array.
    unsigned int size;
    unsigned int keySize;
    unsigned int stashSize;
    unsigned int MAX_FUNCS; // maximum # of hash functions in use.
    // how many times can we try to do an insert before reporting failure?
    short MAX_ATTEMPTS; 

    // Yeah, it's a monster of an initializer.
    HEMI_DEV_CALLABLE_INLINE_MEMBER 
      CuckooSignatureTable(unsigned int * key1, unsigned int * key2, unsigned int * key3, 
                           unsigned long long int * table, unsigned int slots, unsigned int keySlots, unsigned long long int * stash, 
                           unsigned int stashSlots, unsigned int * hash_a, unsigned int * hash_b, 
                           unsigned int hash_count, unsigned int * top, unsigned int * stashTop, 
                           bool* rebuild) 
                           : keyPt1(key1), keyPt2(key2), keyPt3(key3), hash_a(hash_a), hash_b(hash_b), size(slots), stash(stash), table(table), stashSize(stashSlots), MAX_FUNCS(hash_count), top(top), stashTop(stashTop), rebuild(rebuild), MAX_ATTEMPTS(HEMI_CONSTANT(DEFAULT_MAX_ATTEMPTS)), keySize(keySlots) {}
    
    // Copy-constructor that shares the same keytables, but creates a new hash/stash table using the same functions.
    HEMI_DEV_CALLABLE_INLINE_MEMBER 
      CuckooSignatureTable(CuckooSignatureTable parent,  unsigned long long int * table, 
                           unsigned int slots, unsigned long long int * stash, unsigned int stashSlots) :
                           keyPt1(parent.keyPt1), keyPt2(parent.keyPt2), keyPt3(parent.keyPt3), hash_a(parent.hash_a), 
                           hash_b(parent.hash_b), size(slots), stash(stash), table(table), stashSize(stashSlots), 
                           MAX_FUNCS(parent.MAX_FUNCS), top(top), stashTop(stashTop), keySize(parent.keySize),
                           rebuild(rebuild), MAX_ATTEMPTS(HEMI_CONSTANT(DEFAULT_MAX_ATTEMPTS)) {}
    // Copy-constructor that shares the same keytables, but creates a new hash/stash table using (possibly) different functions.
    HEMI_DEV_CALLABLE_INLINE_MEMBER 
      CuckooSignatureTable(CuckooSignatureTable parent,  unsigned long long int * table, 
                           unsigned int slots, unsigned long long int * stash, unsigned int stashSlots, unsigned int * hash_a, unsigned int* hash_b, unsigned int hash_funcs) :
                           keyPt1(parent.keyPt1), keyPt2(parent.keyPt2), keyPt3(parent.keyPt3), hash_a(hash_a), 
                           hash_b(hash_b), size(slots), stash(stash), table(table), stashSize(stashSlots), 
                           MAX_FUNCS(hash_funcs), top(top), stashTop(stashTop), keySize(parent.keySize),
                           rebuild(rebuild), MAX_ATTEMPTS(HEMI_CONSTANT(DEFAULT_MAX_ATTEMPTS)) {}


    HEMI_DEV_CALLABLE_INLINE_MEMBER unsigned int Size() const { return keySize; }

    // Returns a copy of the position in the keypt arrays that we put the K/V
    // values in for use in indexing additional arrays.
    // TODO: Be able to deal with failure from above. 
    HEMI_DEV_CALLABLE_INLINE_MEMBER unsigned int insert(unsigned int key1, unsigned int key2, unsigned int key3) 
    {
      // Check to see if the same key is already in the table. If it is, just
      // return its value without doing anything more.
      unsigned int newpos = hemi::atomicAdd(top, 1);
      assert(newpos < keySize); // make sure we haven't overfilled our key arrays.
      unsigned int result = keySize;
      if ((result = retrieve(key1, key2, key3)) < keySize)
        return result;

      // Compute the signature for this key.
      unsigned int sig = computeSignature(key1, key2, key3);
      
      // Try to put it into the table.
      result = keyInsert(sig, newpos);

      if (result != newpos)
      {
        // TODO: Rebuild table on failure
        printf("Problem inserting %d,%d,%d\n", key1, key2, key3);
        return newpos;
      }

      keyPt1[newpos] = key1;
      keyPt2[newpos] = key2;
      keyPt3[newpos] = key3;

      return result;
    }

    // Returns the value part of table, or size if not found.
    HEMI_DEV_CALLABLE_INLINE_MEMBER unsigned int retrieve(unsigned int key1, unsigned int key2, unsigned int key3) const
    {
      unsigned int sig = computeSignature(key1, key2, key3);
      unsigned int location_0 = hashfunc(sig,0);
      unsigned int location_1 = hashfunc(sig,1);
      unsigned int location_2 = hashfunc(sig,2);
      unsigned int location_3 = hashfunc(sig,3);
      unsigned int location_4 = hashfunc(sig,4);

      TableSlot entry = table[location_0];
      if(!keymatch(key1,key2,key3, entry))
      {
        entry = TableSlot(table[location_1]);
        if(!keymatch(key1,key2,key3, entry))
        {
          entry = TableSlot(table[location_2]);
          if(!keymatch(key1,key2,key3, entry))
          {
            entry =TableSlot(table[location_3]);
            if(!keymatch(key1,key2,key3, entry))
            {
              entry = TableSlot(table[location_4]);
              if(!keymatch(key1,key2,key3, entry))
              {
                // check the stash
                entry = retrieveStash(key1, key2, key3);
                if (entry == HEMI_CONSTANT(EMPTY_SLOT))
                  entry.value = keySize;
              }
            }
          }
        }
      }
      return entry.value;
    }

    // The only important things about these two #s is that they are larger than
    // size and are prime.
    HEMI_DEV_CALLABLE_INLINE_MEMBER unsigned int computeSignature(const unsigned int key1, const unsigned int key2, const unsigned int key3) const
    {
      return (key1*6000011+key2)* 6000023 + key3;
    }
    HEMI_DEV_CALLABLE_INLINE_MEMBER unsigned int hashfunc(unsigned int key, unsigned int n) const 
    {
      return (hash_a[n] * key + hash_b[n]) % HEMI_CONSTANT(lg_prime) % size;
    }
};
typedef CuckooSignatureTable SigTable;


#endif
