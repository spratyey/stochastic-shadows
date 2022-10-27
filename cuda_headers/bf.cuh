#pragma once

#include "common.h"
#include "lcg_random.h"

__device__
void findIndex(int item, int intIdx[NUM_HASH], int bitIdx[NUM_HASH]) {
  uint32_t tmp;
  // for (int j = 0; j < 32 / (NUM_HASH*NUM_LSB); j += 1) {
  // TODO: Make this dynamic
  uint32_t hash = murmur_hash3_mix(item, 5);
  // Split the hash
  for (int i = 0; i < NUM_HASH; i++) {
    tmp = hash & ((1 << NUM_LSB) - 1);
    intIdx[i] = tmp / NUM_BITS;
    bitIdx[i] = tmp - (intIdx[i] * NUM_BITS);
    hash = hash >> NUM_LSB;
  }
  // }
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
