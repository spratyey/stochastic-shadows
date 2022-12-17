#pragma once

#include "owl/common/math/vec.h"
#include "common.cuh"
#include "frostbite.cuh"
#include "utils.cuh"
#include "bvh.cuh"
#include "ltc_utils.cuh"

using namespace owl;


__device__
vec3f ltcMonteCarlo(SurfaceInteraction& si, LCGRand& rng)
{
    vec3f wo_local = normalize(apply_mat(si.to_local, si.wo));
    if (wo_local.z < 0.f)
        return vec3f(0.f);

    vec3f normal_local(0.f, 0.f, 1.f);
    vec3f color(0.0, 0.0, 0.0);

    /* Analytic shading via LTCs */
    vec3f ltc_mat[3], ltc_mat_inv[3];
    float alpha = si.alpha;
    float theta = sphericalTheta(wo_local);

    float amplitude = 1.f;
    fetchLtcMat(alpha, theta, ltc_mat, amplitude);
    matrixInverse(ltc_mat, ltc_mat_inv);

    vec3f iso_frame[3];

    iso_frame[0] = wo_local;
    iso_frame[0].z = 0.f;
    iso_frame[0] = normalize(iso_frame[0]);
    iso_frame[2] = normal_local;
    iso_frame[1] = normalize(cross(iso_frame[2], iso_frame[0]));

    for (int i = 0; i < SAMPLES; i++) {
        vec3f tmpColor = vec3f(0);
        vec2f rand1 = vec2f(lcg_randomf(rng), lcg_randomf(rng));
        vec2f rand2 = vec2f(lcg_randomf(rng), lcg_randomf(rng));

        vec3f lightSample = vec3f(0.f);

        int selectedTriLight;
        float lightSelectionPdf;

        selectFromLBVH(si, selectedTriLight, lightSelectionPdf, rand1, rand2);
        TriLight light = optixLaunchParams.triLights[selectedTriLight];

        tmpColor = integrateOverPolygon(si, ltc_mat, ltc_mat_inv, amplitude, iso_frame, light) / lightSelectionPdf;

        // Make sure there are no negative colors!
        color.x += owl::max(0.f, tmpColor.x);
        color.y += owl::max(0.f, tmpColor.y);
        color.z += owl::max(0.f, tmpColor.z);
    }

    return color / SAMPLES;
}