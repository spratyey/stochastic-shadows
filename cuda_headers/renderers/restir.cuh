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
#if defined(SPATIAL_REUSE)
    vec2i pixelId = owl::getLaunchIndex();
    int fbOfs = pixelId.x + optixLaunchParams.bufferSize.x * pixelId.y;
#endif
    for (int i = 0; i < SAMPLES; i++) {
        Reservoir res(&rng);
        vec3f tmpColor = vec3f(0.0);
        sampleLightRIS<TriLight>(si, optixLaunchParams.triLights, optixLaunchParams.numTriLights, res, rng);

        // Visibility reuse
        float pHat;
        {
            vec3f wi = normalize(res.selP - si.p);
            vec3f wiLocal = normalize(apply_mat(si.to_local, wi));
            vec3f brdf = evaluateBrdf(si.wo_local, wiLocal, si.diffuse, si.alpha);

            ShadowRay ray;
            ray.origin = si.p + 1e-3f * si.n_geom;
            ray.direction = wi;

            ShadowRayData srd;
            owl::traceRay(optixLaunchParams.world, ray, srd);

            if (!(wiLocal.z > 0.f && si.wo_local.z > 0.f && srd.visibility != vec3f(0.f) && dot(-wi, srd.normal) > 0.f)) {
                // Visibility reuse
                res.W = 0.0;
                res.wSum = 0.0;
            }
        }

#if defined(TEMPORAL_REUSE)
        if (optixLaunchParams.accumId > 0 && optixLaunchParams.passId == 0) {
            Reservoir prevRes(&rng);
            prevRes.unpack(optixLaunchParams.resFloatBuffer[fbOfs], optixLaunchParams.resIntBuffer[fbOfs]);

            vec3f wi = normalize(prevRes.selP - si.p);
            vec3f wiLocal = normalize(apply_mat(si.to_local, wi));
            vec3f brdf = evaluateBrdf(si.wo_local, wiLocal, si.diffuse, si.alpha);
            float pHat = length(brdf*optixLaunchParams.triLights[prevRes.selIdx].emit*max(wiLocal.z, 0.0f));

            res.update(prevRes.selIdx, prevRes.selP, pHat * prevRes.W, min(res.M * 20, prevRes.M));
        }
#endif

#if defined(SPATIAL_REUSE)
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

                // Verify that there is some geometry not empty space
                if (optixLaunchParams.depthBuffer[neighFbOfs] < EPS) {
                    continue;
                }

                // Reduce bias due to geometric difference 
                if (abs(optixLaunchParams.depthBuffer[fbOfs] - optixLaunchParams.depthBuffer[neighFbOfs]) > 0.1 * optixLaunchParams.depthBuffer[fbOfs]) {
                    continue;
                }
                if (dot(normalize((vec3f)optixLaunchParams.normalBuffer[fbOfs]), normalize((vec3f)optixLaunchParams.normalBuffer[neighFbOfs])) < 0.906) {
                    continue;
                }

                neighRes.unpack(optixLaunchParams.resFloatBuffer[neighFbOfs], optixLaunchParams.resIntBuffer[neighFbOfs]);

                vec3f wi = normalize(neighRes.selP - si.p);
                vec3f wiLocal = normalize(apply_mat(si.to_local, wi));
                vec3f brdf = evaluateBrdf(si.wo_local, wiLocal, si.diffuse, si.alpha);
                pHat = length(brdf*optixLaunchParams.triLights[neighRes.selIdx].emit)*max(wiLocal.z, 0.0f);
                // neighRes.W = isnan(neighRes.W)? 1e-5:neighRes.W;
                // if (optixLaunchParams.passId == 1) {
                //     print_pixel("%d %f %f %f %d\n", neighRes.selIdx, pHat, neighRes.W, wiLocal.z, neighRes.M);
                // }
                res.update(neighRes.selIdx, neighRes.selP, pHat * neighRes.W, neighRes.M);
            }
        }
#endif
        // Shade pixel
        vec3f wi = normalize(res.selP - si.p);
        vec3f wiLocal = normalize(apply_mat(si.to_local, wi));
        vec3f brdf = evaluateBrdf(si.wo_local, wiLocal, si.diffuse, si.alpha);
        pHat = length(brdf*optixLaunchParams.triLights[res.selIdx].emit*wiLocal.z);

        if (pHat < EPS) {
            res.W = 0.0;
        } else {
            res.W = (1.0 / pHat) * (res.wSum / res.M);
        }
        print_pixel("%f %f %f %d\n", res.wSum, pHat, wiLocal.z, res.M);

        tmpColor = brdf * optixLaunchParams.triLights[res.selIdx].emit * wiLocal.z * res.W;

#ifdef SPATIAL_REUSE
        if (optixLaunchParams.passId == NUM_PASSES) {
            color.x += max(0.0, tmpColor.x);
            color.y += max(0.0, tmpColor.y);
            color.z += max(0.0, tmpColor.z);
        } else {
            color = vec3f(0);
        }
#else
        // Make sure there are no negative colors!
        color.x += max(0.0, tmpColor.x);
        color.y += max(0.0, tmpColor.y);
        color.z += max(0.0, tmpColor.z);

#endif
    }

    // Don't return color for spatial reuse pass
        // print_pixel_exact(500, 500, "%d\n", optixLaunchParams.passId);
    // if (optixLaunchParams.passId != NUM_PASSES) {
    //     return vec3f(0.0);
    // }
    return color / SAMPLES;
}