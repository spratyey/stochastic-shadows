#pragma once

#include "common.h"

struct BST {
    int data = -1;
    int left = -1;
    int right = -1;
};

__device__
float deterministicTraverseLBVH(LightBVH* bvh, int bvhHeight, int rootNodeIdx, SurfaceInteraction& si, vec3f point, int& idx)
{
    float pdf = 1.f;

    int nodeIdx = rootNodeIdx;
    for (int i = 0; i < bvhHeight + 1; i++) {
        LightBVH node = bvh[nodeIdx];

        if (node.left == 0 && node.right == 0) {
            idx = node.primIdx;
            if (node.primCount != 1) {
                pdf *= 1.f / node.primCount;
            }
            break;
        }

        LightBVH leftNode = bvh[node.left];
        LightBVH rightNode = bvh[node.right];

        float leftImp = leftNode.flux / pow(owl::length(leftNode.aabbMid - si.p), 2.f);
        float rightImp = rightNode.flux / pow(owl::length(rightNode.aabbMid - si.p), 2.f);
        float sum = leftImp + rightImp;

        leftImp = leftImp / sum;
        rightImp = rightImp / sum;

        if (point.x >= leftNode.aabbMin.x && point.x <= leftNode.aabbMax.x
            && point.y >= leftNode.aabbMin.y && point.y <= leftNode.aabbMax.y
            && point.z >= leftNode.aabbMin.z && point.z <= leftNode.aabbMax.z) {
            nodeIdx = node.left;
            pdf *= leftImp;
        }
        else if (point.x >= rightNode.aabbMin.x && point.x <= rightNode.aabbMax.x
            && point.y >= rightNode.aabbMin.y && point.y <= rightNode.aabbMax.y
            && point.z >= rightNode.aabbMin.z && point.z <= rightNode.aabbMax.z) {
            nodeIdx = node.right;
            pdf *= rightImp;
        }
        else {
            break;
        }
    }

    return pdf;
}

__device__
void stochasticTraverseLBVH(LightBVH* bvh, int bvhHeight, int rootNodeIdx, SurfaceInteraction& si, int& selectedIdx,
                    float& lightSelectionPdf, vec2f randVec)
{
    selectedIdx = -1;
    lightSelectionPdf = 1.f;

    float r1 = randVec.x;
    float r2 = randVec.y;

    int nodeIdx = rootNodeIdx;
    for (int i = 0; i < bvhHeight + 1; i++) {
        LightBVH node = bvh[nodeIdx];

        // If leaf
        if (node.left == 0 && node.right == 0) {
            if (node.primCount == 1) {
                selectedIdx = node.primIdx;
            }
            else {
                selectedIdx = node.primIdx + round(r1 * (node.primCount-1));
                lightSelectionPdf *= 1.f / node.primCount;
            }

            break;
        }

        LightBVH leftNode = bvh[node.left];
        LightBVH rightNode = bvh[node.right];

        float leftImp = leftNode.flux / pow(owl::length(leftNode.aabbMid - si.p), 2.f);
        float rightImp = rightNode.flux / pow(owl::length(rightNode.aabbMid - si.p), 2.f);
        float sum = leftImp + rightImp;

        leftImp = leftImp / sum;
        rightImp = rightImp / sum;

        float eps = r2;
        if (eps < leftImp) {
            nodeIdx = node.left;
            lightSelectionPdf *= leftImp;
        }
        else {
            nodeIdx = node.right;
            lightSelectionPdf *= rightImp;
        }

        if (r1 < leftImp)
            r1 = r1 / leftImp;
        else
            r1 = (r1 - leftImp) / rightImp;

        if (r2 < leftImp)
            r2 = r2 / leftImp;
        else
            r2 = (r2 - leftImp) / rightImp;
    }
}

__device__ 
void selectFromLBVH(SurfaceInteraction& si, int& selectedIdx, float& lightSelectionPdf, vec2f rand0, vec2f rand1)
{
    // First, traverse the light TLAS and retrive the mesh light
    float lightTlasPdf = 1.f;
    int lightTlasIdx = 0;
    int lightTlasRootNodeIdx = 0;

    stochasticTraverseLBVH(optixLaunchParams.lightTlas, optixLaunchParams.lightTlasHeight, lightTlasRootNodeIdx,
        si, lightTlasIdx, lightTlasPdf, rand0);

    MeshLight meshLight = optixLaunchParams.meshLights[lightTlasIdx];

    // Finally, traverse the light BLAS and get the actual triangle
    float lightBlasPdf = 1.f;
    int lightBlasIdx = 0;
    int lightBlasRootNodeIdx = meshLight.bvhIdx;

    stochasticTraverseLBVH(optixLaunchParams.lightBlas, meshLight.bvhHeight, lightBlasRootNodeIdx,
        si, lightBlasIdx, lightBlasPdf, rand1);

    selectedIdx = lightBlasIdx;
    lightSelectionPdf = lightTlasPdf * lightBlasPdf;
}

__device__ 
void selectFromLBVHSil(SurfaceInteraction& si, int& selectedIdx, float& lightSelectionPdf, vec2f rand0, vec2f rand1)
{
    // First, traverse the light TLAS and retrive the mesh light
    float lightTlasPdf = 1.f;
    int lightTlasIdx = 0;
    int lightTlasRootNodeIdx = 0;

    stochasticTraverseLBVH(optixLaunchParams.lightTlas, optixLaunchParams.lightTlasHeight, lightTlasRootNodeIdx,
        si, lightTlasIdx, lightTlasPdf, rand0);

    selectedIdx = lightTlasIdx;
    lightSelectionPdf = lightTlasPdf;
}

__device__ 
float pdfFromLBVH(SurfaceInteraction& si, vec3f point)
{
    int meshIdx = 0;
    float tlasPdf = deterministicTraverseLBVH(optixLaunchParams.lightTlas, optixLaunchParams.lightTlasHeight, 
        0, si, point, meshIdx);
    
    MeshLight meshLight = optixLaunchParams.meshLights[meshIdx];
    int triIdx = 0;
    float blasPdf = deterministicTraverseLBVH(optixLaunchParams.lightBlas, meshLight.bvhHeight,
        meshLight.bvhIdx, si, point, triIdx);

    return tlasPdf * blasPdf;
}
