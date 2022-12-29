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
void sampleLightRIS(SurfaceInteraction &si, LightType *lights, int lightCount, Reservoir &res, LCGRand &rng) {
    vec3f selP;
    int selIdx;
    float selPdf;
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
        float w = pHat / max(selPdf, EPS);
        res.update(selIdx, selP, w, 1);
    }
}