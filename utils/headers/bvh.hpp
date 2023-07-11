#pragma once

#include "types.hpp"
#include <queue>

float evaluateSAHForLightBVH(LightBVH& node, std::vector<TriLight>& primitives, int axis, float pos);

int getLightBVHHeight(uint32_t nodeIdx, std::vector<LightBVH>& bvh);

template <typename T>
void updateLightBVHNodeBounds(uint32_t nodeIdx, std::vector<LightBVH> &bvh, std::vector<T> &primitives)
{
    bvh[nodeIdx].aabbMax = vec3f(-1e30f);
    bvh[nodeIdx].aabbMin = vec3f(1e30f);

    for (uint32_t i = bvh[nodeIdx].primIdx; i < bvh[nodeIdx].primIdx + bvh[nodeIdx].primCount; i++) {
        T& prim = primitives[i];

        bvh[nodeIdx].aabbMin = owl::min(bvh[nodeIdx].aabbMin, prim.aabbMin);
        bvh[nodeIdx].aabbMax = owl::max(bvh[nodeIdx].aabbMax, prim.aabbMax);
        bvh[nodeIdx].flux += prim.flux;
    }

    bvh[nodeIdx].aabbMid = (bvh[nodeIdx].aabbMin + bvh[nodeIdx].aabbMax) * 0.5f;
    bvh[nodeIdx].flux /= bvh[nodeIdx].primCount;
}

template <typename T>
void subdivideLightBVH(uint32_t nodeIdx, std::vector<LightBVH>& bvh, std::vector<T>& primitives)
{
    // TODO: Make this more elegant
    if (bvh[nodeIdx].primCount <= 1) {
        bvh[nodeIdx].flux = 0.f;
        for (int z = bvh[nodeIdx].primIdx; z < bvh[nodeIdx].primIdx + bvh[nodeIdx].primCount; z++) {
            bvh[nodeIdx].flux += primitives[z].flux;
        }

        bvh[nodeIdx].flux /= bvh[nodeIdx].primCount;
    
        return;
    }
    
    vec3f extent = bvh[nodeIdx].aabbMax - bvh[nodeIdx].aabbMin;

    int axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z > extent[axis]) axis = 2;
    float splitPos = bvh[nodeIdx].aabbMin[axis] + extent[axis] * 0.5f;

    int i = bvh[nodeIdx].primIdx;
    int j = i + bvh[nodeIdx].primCount - 1;
    while (i <= j) {
        if (primitives[i].cg[axis] < splitPos) {
            i++;
        }
        else {
            T iItem = primitives[i];
            T jItem = primitives[j];

            primitives[i] = jItem;
            primitives[j] = iItem;

            j--;
        }
    }

    int leftCount = i - bvh[nodeIdx].primIdx;
    if (leftCount == 0 || leftCount == bvh[nodeIdx].primCount) {
        bvh[nodeIdx].flux = 0.f;
        for (int z = bvh[nodeIdx].primIdx; z < bvh[nodeIdx].primIdx + bvh[nodeIdx].primCount; z++) {
            bvh[nodeIdx].flux += primitives[z].flux;
        }

        bvh[nodeIdx].flux /= bvh[nodeIdx].primCount;

        std::cout << bvh[nodeIdx].primCount << std::endl;
        return;
    }

    bvh[nodeIdx].left = bvh.size();
    LightBVH leftNode;
    leftNode.primCount = leftCount;
    leftNode.primIdx = bvh[nodeIdx].primIdx;
    bvh.push_back(leftNode);

    bvh[nodeIdx].right = bvh.size();
    LightBVH rightNode;
    rightNode.primCount = bvh[nodeIdx].primCount - leftCount;
    rightNode.primIdx = i;
    bvh.push_back(rightNode);

    bvh[nodeIdx].primCount = 0;

    updateLightBVHNodeBounds<T>(bvh[nodeIdx].left, bvh, primitives);
    updateLightBVHNodeBounds<T>(bvh[nodeIdx].right, bvh, primitives);

    subdivideLightBVH<T>(bvh[nodeIdx].left, bvh, primitives);
    subdivideLightBVH<T>(bvh[nodeIdx].right, bvh, primitives);

    bvh[nodeIdx].flux = (bvh[bvh[nodeIdx].left].flux + bvh[bvh[nodeIdx].right].flux) / 2.0f;
}

void reorderLightBVH(std::vector<LightBVH> &input, std::vector<LightBVH> &output);