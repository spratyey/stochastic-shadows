#pragma once

#include "bf.cuh"
#include "constants.cuh"

#ifdef USE_BLOOM
class Set {
    private:
        unsigned int bf[NUM_BITS];
    
    public:
        __device__
        Set() {
            initBF(bf);
        }

        __device__
        void insert(int item) {
            insertBF(bf, item);
        }

        __device__
        bool exists(int item) {
            return queryBF(bf, item);
        }
};
#else
class Set {
    private:
        int selectedIdx[MAX_LTC_LIGHTS];
        int insertedCount = 0;
    
    public:
        __device__
        Set() {
            for (int i = 0; i < MAX_LTC_LIGHTS; i += 1) {
                selectedIdx[i] = -1;
            }
        }

        __device__
        void insert(int item) {
            selectedIdx[insertedCount] = item;
            insertedCount += 1;
        }

        __device__
        bool exists(int item) {
            for (int i = 0; i < insertedCount; i += 1) {
                if (selectedIdx[i] == item) {
                    return true;
                }
            }
            return false;
        }
};
#endif