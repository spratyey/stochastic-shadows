#include <cstdlib>
#include <iostream>
#include <cuda.h>
#include "cuda_headers/lcg_random.cuh"

__device__
bool insertBF(unsigned int *bf, int item, int filter_size, int num_hashes) {
  int intIdx, bitIdx;
  int lsb = log2f(filter_size+1);
  int size = (sizeof(unsigned int)*8);
  bool found = true;
  int tmp;
  uint32_t hash = murmur_hash3_finalize(item);
  for (int i = 0; i < ceilf((float)(num_hashes*lsb) / 32); i += 1) {
    hash = murmur_hash3_mix(hash, i);
    // Split the hash
    for (int j = 0; j < 32 / lsb; j++) {
      tmp = hash & ((1 << lsb) - 1);
      intIdx = tmp / size;
      bitIdx = tmp - (intIdx * size);
      hash = hash >> lsb;
      found &= ((bf[intIdx] >> bitIdx) & 1);
      bf[intIdx] |= (1 << bitIdx);
    }
  }

  return found;
}

__global__
void test_bf(int filter_size, int num_hashes, int num_insert, int *collisions) {
  extern __shared__ unsigned int bf[];
  for (int i = 0; i < filter_size / 32; i++) {
    bf[i] = 0;
  }
  *collisions = 0;
  for (int i = 0; i < num_insert; i++) {
    bool found = insertBF(bf, murmur_hash3_finalize(i), filter_size, num_hashes);
    if (found) {
      *collisions += 1;
    }
  }
}

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "USAGE: test_bf <filter_size> <num_hashes> <num_insert>";
  }

  // TODO: Check if these are integers
  int filter_size = atoi(argv[1]);
  int num_hashes = atoi(argv[2]);
  int num_insert = atoi(argv[3]);
  int collisions;

  int *dev_collisions;
  cudaMalloc((void**)&dev_collisions, sizeof(int));

  test_bf<<<1,1,(filter_size / 32) * sizeof(unsigned int)>>>(filter_size, num_hashes, num_insert, dev_collisions);

  cudaMemcpy(&collisions, dev_collisions, sizeof(int), cudaMemcpyDeviceToHost);

  std::cout << "Collisions: " << collisions << std::endl;
  std::cout << "False Positive Rate: " << (float)collisions / (float)num_insert * 100.0 << "%" << std::endl;

  cudaFree(&dev_collisions);

}
