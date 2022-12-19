#pragma once

#include "owl/common/math/vec.h"
#include "samplers/ris.cuh"
#include "common.cuh"
#include "lcg_random.cuh"
#include "types.hpp"

using namespace owl;

__device__
vec3f estimateDirectLightingReSTIR(SurfaceInteraction& si, LCGRand& rng) {
    vec3f color = vec3f(0.f);
    int selIdx;
    float selPdf;
    vec3f selP; 
    for (int i = 0; i < SAMPLES; i++) {
        vec3f tmpColor = vec3f(0.0);
        sampleLightRIS<TriLight>(si, optixLaunchParams.triLights, optixLaunchParams.numTriLights, selIdx, selPdf, selP, rng);

        vec3f wi = normalize(selP - si.p);
        vec3f wiLocal = normalize(apply_mat(si.to_local, wi));

        // Shoot shadow ray
        ShadowRay ray;
        ray.origin = si.p + 1e-3f * si.n_geom;
        ray.direction = wi;

        ShadowRayData srd;
        owl::traceRay(optixLaunchParams.world, ray, srd);

        if (wiLocal.z > 0.f && si.wo_local.z > 0.f && srd.visibility != vec3f(0.f) && dot(-wi, srd.normal) > 0.f) {
            vec3f brdf = evaluateBrdf(si.wo_local, wiLocal, si.diffuse, si.alpha);
            float pHat = length(brdf*srd.emit*wiLocal.z);

            tmpColor += ((brdf*srd.emit*wiLocal.z) / pHat) * (selPdf / (float)NUM_PROPOSALS);
        }

        // Make sure there are no negative colors!
        color.x += max(0.0, tmpColor.x);
        color.y += max(0.0, tmpColor.y);
        color.z += max(0.0, tmpColor.z);
    }

    return color / SAMPLES;
}