#pragma once

#include "owl/common/math/vec.h"
#include "lcg_random.cuh"

using namespace std;

class Reservoir {
    private:
        LCGRand *rng;

    public:
        int selIdx;
        vec3f selP;
        float wSum;
        float W;
        int samples;

        __device__
        Reservoir(LCGRand *rng) : samples(0), wSum(0.0), selIdx(0), rng(rng) {}

        __device__ 
        void update(int sample, vec3f &p, float w) {
            wSum += w;
            samples += 1;
            if (lcg_randomf(*rng) < w / wSum || wSum < EPS) {
                selIdx = sample;
                selP = p;
            }
        }

        __device__
        void pack(float4 &floatBuffer, int2 &intBuffer) {
            floatBuffer.x = selP.x;
            floatBuffer.y = selP.y;
            floatBuffer.z = selP.z;
            floatBuffer.w = W;
            intBuffer.x = selIdx;
            intBuffer.y = samples;
        }

        __device__
        void unpack(float4 &floatBuffer, int2 &intBuffer) {
            selP.x = floatBuffer.x;
            selP.y = floatBuffer.y;
            selP.z = floatBuffer.z;
            W = floatBuffer.w;
            selIdx = intBuffer.x;
            samples = intBuffer.y;
        }
};