#pragma once

#include "owl/common/math/vec.h"
#include "lcg_random.cuh"

using namespace std;

class Reservoir {
    private:
        LCGRand *rng;

    public:
        unsigned int selIdx;
        vec3f selP;
        float wSum;
        float W;
        unsigned int M;

        __device__
        Reservoir(LCGRand *rng) : M(0), wSum(0.0), selIdx(0), rng(rng), selP(vec3f(0))  {}

        __device__ 
        void update(int idx, vec3f &p, float w, int samples) {
            wSum += w*samples;
            M += samples;
            if (lcg_randomf(*rng) < w / max(wSum, 1e-3)) {
                selIdx = idx;
                selP = p;
            }
        }

        __device__
        void pack(float4 &floatBuffer, uint2 &intBuffer) {
            floatBuffer.x = selP.x;
            floatBuffer.y = selP.y;
            floatBuffer.z = selP.z;
            floatBuffer.w = W;
            intBuffer.x = selIdx;
            intBuffer.y = M;
        }

        __device__
        void unpack(float4 &floatBuffer, uint2 &intBuffer) {
            selP.x = floatBuffer.x;
            selP.y = floatBuffer.y;
            selP.z = floatBuffer.z;
            W = floatBuffer.w;
            selIdx = intBuffer.x;
            M = intBuffer.y;
        }
};