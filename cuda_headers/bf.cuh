#pragma once

#include "lcg_random.cuh"
#include "constants.cuh"

__device__
void findIndex(int item, int intIdx[NUM_HASH], int bitIdx[NUM_HASH]) {
  uint32_t tmp;
  uint32_t hash = murmur_hash3_finalize(murmur_hash3_finalize(item));
  int hash_count = 0;
#pragma unroll
  for (int i = 0; i < ceilf((float)(NUM_HASH*NUM_LSB) / 32); i += 1) {
    hash = murmur_hash3_mix(hash, i);
    // Split the hash
    for (int j = 0; j < 32 / NUM_LSB && hash_count < NUM_HASH; j++) {
      tmp = hash & ((1 << NUM_LSB) - 1);
      intIdx[hash_count] = tmp / 32;
      bitIdx[hash_count] = tmp - (intIdx[hash_count] * 32);
      hash = hash >> NUM_LSB;
      hash_count += 1;
    }
  }
}

__device__
void initBF(unsigned int *bf) {
  for (int i = 0; i < NUM_BITS; i++) {
    bf[i] = 0;
  }
}

__device__
void insertBF(unsigned int *bf, int item) {
  int intIdx[NUM_HASH];
  int bitIdx[NUM_HASH];
  findIndex(item, intIdx, bitIdx);
  for (int i = 0; i < NUM_HASH; i++) {
    bf[intIdx[i]] |= (1 << bitIdx[i]);
  }
}

__device__
bool queryBF(unsigned int *bf, int item) {
  bool found = true;
  int intIdx[NUM_HASH];
  int bitIdx[NUM_HASH];
  findIndex(item, intIdx, bitIdx);
  for (int i = 0; i < NUM_HASH; i++) {
    found &= ((bf[intIdx[i]] >> bitIdx[i]) & 1);
    if (!found) {
      break;
    }
  }

  return found;
}
