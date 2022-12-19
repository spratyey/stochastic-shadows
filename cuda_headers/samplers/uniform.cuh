#pragma once

#include "owl/common/math/vec.h"
#include "types.hpp"
#include "lcg_random.cuh"
#include "utils.cuh"

using namespace owl;

__device__
void sampleLightUniform(TriLight *lights, int lightCount, int &selIdx, float &selPdf, vec3f &selP, LCGRand &rng) {
    vec3f rand = vec3f(lcg_randomf(rng), lcg_randomf(rng), lcg_randomf(rng));
    selIdx = lightCount * rand.x;
    selPdf = 1.0 / (float)lightCount;

    TriLight triLight = lights[selIdx];
    vec3f lv1 = triLight.v1;
    vec3f lv2 = triLight.v2;
    vec3f lv3 = triLight.v3;
    selP = samplePointOnTriangle(lv1, lv2, lv3, rand.y, rand.z);
    selPdf *= 1 / triLight.area;
}