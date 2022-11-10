#pragma once

#include "owl/common/math/vec.h"
#include "common.cuh"
#include "utils.cuh"
#include "types.hpp"

using namespace owl;

__device__
BSPNode getSilEdges(int lightIdx, vec3f &p) {
    MeshLight light = optixLaunchParams.meshLights[lightIdx];
    int startNode = light.spans.bspNodeSpan.x;
    BSPNode node = optixLaunchParams.bsp[startNode + light.bspRoot];
    while (node.left >= 0 && node.right >= 0) {
        float dist = getPlanePointDist(p, node.plane);
        if (dist > 0) {
            node = optixLaunchParams.bsp[startNode + node.left];
        } else {
            node = optixLaunchParams.bsp[startNode + node.right];
        }
    }
    return node;
}