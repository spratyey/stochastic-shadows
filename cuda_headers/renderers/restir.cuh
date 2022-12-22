#pragma once

#include "owl/common/math/vec.h"
#include "samplers/ris.cuh"
#include "common.cuh"
#include "lcg_random.cuh"
#include "types.hpp"

using namespace owl;

__device__
vec3f estimateDirectLightingReSTIR(SurfaceInteraction& si, LCGRand& rng, Reservoir &res) {
    vec3f color = vec3f(0.f);
#if defined(USE_RESERVOIRS) && defined(SPATIAL_REUSE)
    vec2i pixelId = owl::getLaunchIndex();
    int fbOfs = pixelId.x + optixLaunchParams.bufferSize.x * pixelId.y;
#endif
    for (int i = 0; i < SAMPLES; i++) {
        vec3f tmpColor = vec3f(0.0);
        sampleLightRIS<TriLight>(si, optixLaunchParams.triLights, optixLaunchParams.numTriLights, res, rng);

#if defined(USE_RESERVOIRS) && defined(SPATIAL_REUSE)
        if (optixLaunchParams.passId > 0) {
            Reservoir neighRes(&rng);
            for (int i = 0; i < SPATIAL_SAMPLES; i += 1) {
                float radius = KERNEL_SIZE * lcg_randomf(rng);
                float angle = 2.0f * PI * lcg_randomf(rng);
                int neighX = pixelId.x + radius * cos(angle);
                int neighY = pixelId.y + radius * sin(angle);
                
                // Boundary check
                if (neighX < 0 || neighX >= optixLaunchParams.bufferSize.x ||
                    neighY < 0 || neighY >= optixLaunchParams.bufferSize.y) {
                        continue;
                    }

                int neighFbOfs = neighX + optixLaunchParams.bufferSize.x * neighY;
                
                // Reduce bias due to geometric difference 
                if (abs(optixLaunchParams.depthBuffer[fbOfs] - optixLaunchParams.depthBuffer[neighFbOfs]) > 0.1 * optixLaunchParams.depthBuffer[fbOfs]) {
                    continue;
                }
                if (dot((vec3f)optixLaunchParams.normalBuffer[fbOfs], (vec3f)optixLaunchParams.normalBuffer[neighFbOfs]) < 0.906) {
                    continue;
                }

                neighRes.unpack(optixLaunchParams.resFloatBuffer[neighFbOfs], optixLaunchParams.resIntBuffer[neighFbOfs]);

                vec3f wi = normalize(neighRes.selP - si.p);
                vec3f wiLocal = normalize(apply_mat(si.to_local, wi));
                vec3f brdf = evaluateBrdf(si.wo_local, wiLocal, si.diffuse, si.alpha);
                float pHat = length(brdf*optixLaunchParams.triLights[neighRes.selIdx].emit*max(wiLocal.z, 0.0f));

                res.update(neighRes.selIdx, neighRes.selP, pHat * neighRes.W * neighRes.samples);
                res.samples += neighRes.samples;
            }
        }
#endif

        vec3f wi = normalize(res.selP - si.p);
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

            res.W = (1.0 / pHat) * (res.wSum / res.samples);

            tmpColor += brdf * srd.emit * wiLocal.z * res.W;
        } else {
            // Visibility reuse
            res.W = 0.0;
            res.wSum = 0.0;
        }

        // Make sure there are no negative colors!
        color.x += max(0.0, tmpColor.x);
        color.y += max(0.0, tmpColor.y);
        color.z += max(0.0, tmpColor.z);
    }

    return color / SAMPLES;
}