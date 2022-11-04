#pragma once

#include "common.cuh"
#include "utils.cuh"

__device__
vec3f colorEdges(SurfaceInteraction& si, RadianceRay ray)
{
  vec3f p = si.p;
  vec3f camPos = optixLaunchParams.camera.pos;
  vec3f onb[3];
  vec3f unused[3];
  orthonormalBasis(ray.direction, onb, unused);

  int edgeIdx = -1;
  for (int i = 0; i < optixLaunchParams.numMeshLights; i += 1) {
    int edgeStartIdx = optixLaunchParams.meshLights[i].edgeStartIdx;
    int edgeCount = optixLaunchParams.meshLights[i].edgeCount;
    for (int j = edgeStartIdx; j < edgeStartIdx + edgeCount; j += 1) {
      LightEdge edge = optixLaunchParams.lightEdges[j];
      float perpDist = owl::length(owl::cross(edge.v1 - p, edge.v2 - edge.v1)) / owl::length(edge.v2 - edge.v1);
      if (perpDist < 0.1) {
        edgeIdx = j;
        break;
      }
    }
  }

  if (edgeIdx >= 0) {
    LightEdge edge = optixLaunchParams.lightEdges[edgeIdx];
    bool isSil;
    if (edge.adjFaceCount == 2) {
      isSil = owl::dot(edge.n1, edge.v1 - camPos) * owl::dot(edge.n2, edge.v2 - camPos) < 0;
    } else {
      isSil = true;
    }
    if (isSil) {
      float edgeLen = owl::length(edge.v1 - edge.v2);
      float v1Len = owl::length(edge.v1 - si.p);
      float v2Len = owl::length(edge.v2 - si.p);

      bool toFlip = shouldFlip(edge, camPos);

      // Red -> v1, Green -> v2
      vec3f c1 = toFlip ? vec3f(1, 0, 0) : vec3f(0, 1, 0);
      vec3f c2 = vec3f(1, 1, 0) - c1;

      return v1Len / edgeLen * c1 + v2Len / edgeLen * c2;
    } else {
      return vec3f(0, 0, 1);
    }
  }

  return si.diffuse;
}
