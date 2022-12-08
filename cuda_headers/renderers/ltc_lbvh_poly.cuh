#pragma once

#include "common.cuh"
#include "utils.cuh"
#include "bvh.cuh"
#include "set.cuh"
#include "ltc_utils.cuh"
#include "lcg_random.cuh"
#include "constants.cuh"

__device__
vec3f ltcDirectLightingLBVHPoly(SurfaceInteraction& si, LCGRand& rng)
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
    iso_frame[1] = normalize(cross(iso_frame[2], iso_frame[0]));

    int selectedIdx[MAX_LTC_LIGHTS] = { -1 };
    int selectedEnd = 0;

#ifdef SIL
    BSPNode sils[MAX_LTC_LIGHTS];
#endif

    int ridx = 0;
    vec3f color(0.f, 0.f, 0.f);

    Set selectedSet;
#ifdef REJECTION_SAMPLING
    for (int i = 0; i < MAX_LTC_LIGHTS * 2; i++) {
        if (selectedEnd == optixLaunchParams.numMeshLights || selectedEnd == MAX_LTC_LIGHTS) {
            break;
        }

        rand0 = vec2f(lcg_randomf(rng), lcg_randomf(rng));

        ridx = 0;
        selectFromLBVHSil(si, ridx, rand0);

        bool found = selectedSet.exists(ridx);

        if (!found) {
            selectedSet.insert(ridx);
            selectedIdx[selectedEnd++] = ridx;
        }
    }
#else
    // Use better BVH sampling
    float unused;
    int elemsChosen = 0;
    for (int i = 0; i < MAX_LTC_LIGHTS; i++) {
        if (selectedEnd == optixLaunchParams.numMeshLights || elemsChosen > MAX_ELEMS) {
            break;
        }

        rand0 = vec2f(lcg_randomf(rng), lcg_randomf(rng));

        ridx = 0;
        stochasticTraverseLBVHNoDup(optixLaunchParams.lightTlas, optixLaunchParams.lightTlasHeight, 0, si, &selectedSet, ridx, unused, rand0);
        // Use below for profiling
        // stochasticTraverseLBVH(optixLaunchParams.lightTlas, optixLaunchParams.lightTlasHeight, 0, si, ridx, unused, rand0);
        selectedIdx[selectedEnd] = ridx;
#ifdef SIL
        BSPNode node = getSilEdges(ridx, si.p);
        elemsChosen += node.silSpan.y - node.silSpan.x;
        sils[selectedEnd] = node;
#else
        MeshLight light = optixLaunchParams.meshLights[selectedIdx[i]];
        elemsChosen += light.triCount;
#endif
        selectedEnd += 1;
    }
#endif // REJECTION_SAMPLING

    print_pixel("%d %d\n", selectedEnd, elemsChosen);
    for (int i = 0; i < selectedEnd; i++) {
        print_pixel("%d ", selectedIdx[i]);
// #ifdef SIL
//         color += integrateOverPolyhedron(si, ltc_mat, ltc_mat_inv, amplitude, iso_frame, sils[i], selectedIdx[i]);
// #else
//         MeshLight light = optixLaunchParams.meshLights[selectedIdx[i]];
//         for (int j = light.triIdx; j < light.triIdx + light.triCount; j += 1) {
//             color += integrateOverPolygon(si, ltc_mat, ltc_mat_inv, amplitude, iso_frame,
//                 optixLaunchParams.triLights[j]);
//         }
// #endif
        // Use this for profiling
        color += optixLaunchParams.meshLights[selectedIdx[i]].avgEmit;
    }
    print_pixel("\n");

    return color;
}
