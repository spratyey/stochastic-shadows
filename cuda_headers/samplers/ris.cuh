#pragma once

#include "owl/common/math/vec.h"
#include "reservoir.cuh"
#include "lcg_random.cuh"
#include "constants.cuh"
#include "common.cuh"
#include "samplers/uniform.cuh"

using namespace owl;

template<typename LightType>
__device__
void sampleLightRIS(SurfaceInteraction &si, LightType *lights, int lightCount, int &selIdx, float &selPdf, vec3f &selP, LCGRand &rng) {
#ifdef USE_RESERVOIRS
    Reservoir res(&rng);
#else
    float wSum = 0;
    vec3f proposalsP[NUM_PROPOSALS];
    int proposalsIdx[NUM_PROPOSALS];
    float weights[NUM_PROPOSALS];
#endif

    for (int i = 0; i < NUM_PROPOSALS; i++) {
        sampleLightUniform(lights, lightCount, selIdx, selPdf, selP, rng);

        vec3f wi = normalize(selP - si.p);
        vec3f wiLocal = normalize(apply_mat(si.to_local, wi));

        // Convert from angle measure to area measure
        float xmy = pow(length(selP - si.p), 2.f);
        float lDotWi = abs(dot(lights[selIdx].normal, -wi));
        selPdf *= (xmy / lDotWi);

        vec3f brdf = evaluateBrdf(si.wo_local, wiLocal, si.diffuse, si.alpha);
        float pHat = length(brdf*lights[selIdx].emit*max(wiLocal.z, 0.0f));
        float w = pHat / selPdf;
#ifdef USE_RESERVOIRS
        res.update(selIdx, selP, w);
#else
        wSum += w;
        proposalsP[i] = selP;
        proposalsIdx[i] = selIdx;
        weights[i] = w;
#endif
    }

#ifdef USE_RESERVOIRS
   selIdx = res.selIdx; 
   selPdf = res.wSum;
   selP = res.selP;
#else
    // Generate normalized CDF
    for (int i = 0; i < NUM_PROPOSALS; i++) {
        weights[i] /= wSum;
        if (i > 0) {
            weights[i] += weights[i-1];
        }
    }

    // Choose sample
    selPdf = wSum;
    float rand = lcg_randomf(rng);
    if (rand <= weights[0]) {
        selP = proposalsP[0];
        selIdx = proposalsIdx[0];
    }
    for (int i = 1; i < NUM_PROPOSALS; i++) {
        if (weights[i-1] < rand && rand <= weights[i]) {
            selP = proposalsP[i];
            selIdx = proposalsIdx[i];
            break;
        }
    }
#endif
}