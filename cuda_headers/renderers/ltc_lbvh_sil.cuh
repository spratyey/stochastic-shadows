#pragma once

#include "common.cuh"
#include "utils.cuh"
#include "bvh.cuh"
#include "bf.cuh"
#include "ltc_utils.cuh"
#include "lcg_random.cuh"
#include "constants.cuh"

__device__
vec3f ltcDirectLightingLBVHSil(SurfaceInteraction& si, LCGRand& rng, bool shouldPrint)
{
    vec3f normal_local(0.f, 0.f, 1.f);

    vec2f rand0(lcg_randomf(rng), lcg_randomf(rng));
    vec2f rand1(lcg_randomf(rng), lcg_randomf(rng));

    // Backface
    if (si.wo_local.z < 0.f)
        return vec3f(0.f);

    /* Analytic shading via LTCs */
    vec3f ltc_mat[3], ltc_mat_inv[3];
    float alpha = si.alpha;
    float theta = sphericalTheta(si.wo_local);

    float amplitude = 1.f;
    fetchLtcMat(alpha, theta, ltc_mat, amplitude);
    matrixInverse(ltc_mat, ltc_mat_inv);

    vec3f iso_frame[3];

    iso_frame[0] = si.wo_local;
    iso_frame[0].z = 0.f;
    iso_frame[0] = normalize(iso_frame[0]);
    iso_frame[2] = normal_local;
    iso_frame[1] = normalize(owl::cross(iso_frame[2], iso_frame[0]));
    
#ifdef USE_BLOOM
    unsigned int bf[NUM_BITS] = { 0 };
    initBF(bf);
    int selectedIdx[MAX_LTC_LIGHTS * 2] = { -1 };
#else
    int selectedIdx[MAX_LTC_LIGHTS * 2] = { -1 };
#endif
    int selectedEnd = 0;

    int ridx = 0;
    float rpdf = 0.f;

    selectFromLBVHSil(si, ridx, rpdf, rand0, rand1);
#ifdef USE_BLOOM
    insertBF(bf, ridx);
#endif
    selectedIdx[selectedEnd++] = ridx;

    vec3f color(0.f, 0.f, 0.f);
    for (int i = 0; i < MAX_LTC_LIGHTS * 2; i++) {
        if (selectedEnd == optixLaunchParams.numMeshLights || selectedEnd == MAX_LTC_LIGHTS)
            break;

        rand0 = vec2f(lcg_randomf(rng), lcg_randomf(rng));
        rand1 = vec2f(lcg_randomf(rng), lcg_randomf(rng));

        ridx = 0;
        rpdf = 0.f;
        selectFromLBVHSil(si, ridx, rpdf, rand0, rand1);

        bool found = false;
#ifdef USE_BLOOM
        found = queryBF(bf, ridx);
#else
        for (int j = 0; j < selectedEnd; j++) {
            if (selectedIdx[j] == ridx) {
                found = true;
                break;
            }
        }
#endif

        if (!found) {
          if (shouldPrint) printf("%d ", ridx);
#ifdef USE_BLOOM
          insertBF(bf, ridx);
          selectedIdx[selectedEnd++] = ridx;
            color += integrateOverPolyhedron(si, ltc_mat, ltc_mat_inv, amplitude, iso_frame, ridx, shouldPrint);
#else
          selectedIdx[selectedEnd++] = ridx;
#endif
        }
    }
    if (shouldPrint) printf("\n");

#ifndef BLOOM
    for (int i = 0; i < selectedEnd; i++) {
        color += integrateOverPolyhedron(si, ltc_mat, ltc_mat_inv, amplitude, iso_frame, selectedIdx[i], shouldPrint);
    }
#endif

    return color;
}
