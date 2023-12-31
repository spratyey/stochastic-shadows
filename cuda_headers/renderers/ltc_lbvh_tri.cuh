#pragma once

#include "common.cuh"
#include "utils.cuh"
#include "bvh.cuh"
#include "bf.cuh"
#include "ltc_utils.cuh"
#include "lcg_random.cuh"
#include "sil_utils.cuh"
#include "constants.cuh"


__device__
vec3f ltcDirectLightingLBVHTri(SurfaceInteraction& si, LCGRand& rng) {
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
#else
    int selectedIdx[MAX_LTC_LIGHTS * 2] = { -1 };
#endif
    int selectedEnd = 0;

    int ridx = 0;
    float _rpdf = 0.f;

    selectFromLBVH(si, ridx, _rpdf, rand0, rand1);
#ifdef USE_BLOOM
    insertBF(bf, ridx);
#else
    selectedIdx[selectedEnd++] = ridx;
#endif

    vec3f color(0.f, 0.f, 0.f);

    for (int i = 0; i < MAX_LTC_LIGHTS * 2; i++) {
        if (selectedEnd == optixLaunchParams.numMeshLights || selectedEnd == MAX_LTC_LIGHTS)
            break;

        rand0 = vec2f(lcg_randomf(rng), lcg_randomf(rng));
        rand1 = vec2f(lcg_randomf(rng), lcg_randomf(rng));

        selectFromLBVH(si, ridx, _rpdf, rand0, rand1);

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
#ifdef USE_BLOOM
            insertBF(bf, ridx);
            color += integrateOverPolygon(si, ltc_mat, ltc_mat_inv, amplitude, iso_frame, optixLaunchParams.triLights[ridx]);
            selectedEnd++;
#else
            selectedIdx[selectedEnd++] = ridx;
#endif
        }
    }

#ifndef USE_BLOOM
    for (int i = 0; i < selectedEnd; i++) {
        color += integrateOverPolygon(si, ltc_mat, ltc_mat_inv, amplitude, iso_frame,
            optixLaunchParams.triLights[selectedIdx[i]]);
    }
#endif

    return color;
}
