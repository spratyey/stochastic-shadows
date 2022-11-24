#pragma once

#include "constants.cuh"
#include "owl/common/math/vec.h"

using namespace owl;

__device__
void sampleLights(int chosenLights[MAX_LTC_LIGHTS]) {
#ifdef USE_BLOOM
    unsigned int bf[NUM_BITS] = { 0 };
    initBF(bf);
#endif
    int selectedEnd = 0;

    int ridx = 0;

    selectFromLBVHSil(si, ridx, rand0, rand1);
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
        selectFromLBVHSil(si, ridx, rand0, rand1);

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
}