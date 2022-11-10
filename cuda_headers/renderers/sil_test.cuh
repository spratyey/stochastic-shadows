#pragma once

#include "owl/common/math/vec.h"
#include "common.cuh"
#include "utils.cuh"
#include "sil_utils.cuh"
#include "constants.cuh"

using namespace owl;

__device__
vec3f colorEdges(SurfaceInteraction& si, RadianceRay ray, bool shouldPrint) {
  vec3f p = si.p;
  vec3f camPos = optixLaunchParams.camera.pos;
  vec3f onb[3];
  vec3f unused[3];
  orthonormalBasis(ray.direction, onb, unused);

  // 374, 186
  const vec2i pixelId = owl::getLaunchIndex();

  int edgeIdx = -1;
  int lightIdx = -1;
  for (int i = 0; i < optixLaunchParams.numMeshLights; i += 1) {
    int edgeStartIdx = optixLaunchParams.meshLights[i].spans.edgeSpan.x;
    int edgeEndIdx = optixLaunchParams.meshLights[i].spans.edgeSpan.y;
    for (int j = edgeStartIdx; j < edgeEndIdx; j += 1) {
      LightEdge edge = optixLaunchParams.lightEdges[j];
      float perpDist = length(cross(edge.v1 - p, edge.v2 - edge.v1)) / length(edge.v2 - edge.v1);
      if (perpDist < 0.1) {
        edgeIdx = j;
        lightIdx = i;
        break;
      }
    }
  }

  if (shouldPrint) printf("%d %d\n", edgeIdx, lightIdx);

  if (lightIdx >= 0) {
    bool isSil = false;
    bool toFlip;
    LightEdge edge = optixLaunchParams.lightEdges[edgeIdx];
#ifdef BSP_SIL
    BSPNode node = getSilEdges(lightIdx, camPos);
    vec2i silSpan = node.silSpan;
    int edgeCount = optixLaunchParams.meshLights[lightIdx].spans.edgeSpan.y - optixLaunchParams.meshLights[lightIdx].spans.edgeSpan.x;
    int edgeStartIdx = optixLaunchParams.meshLights[lightIdx].spans.edgeSpan.x;
    toFlip = false;
    if (shouldPrint) printf("%d\n", edgeCount);
    for (int i = silSpan.x; i < silSpan.y; i += 1) {
      int silIdx = optixLaunchParams.silhouettes[i];
      if (abs(silIdx) == edgeIdx - edgeStartIdx || silIdx - edgeCount == edgeIdx - edgeStartIdx) {
        toFlip = silIdx < 0 || silIdx == edgeCount;
        isSil = true;
        break;
      }
    }
#else
    if (edge.adjFaceCount == 2) {
      isSil = owl::dot(edge.n1, edge.v1 - camPos) * owl::dot(edge.n2, edge.v2 - camPos) < 0;
    } else {
      isSil = true;
    }

    if (isSil) {
      bool toFlip = shouldFlip(edge, camPos);
    }
#endif

    if (isSil) {
      float edgeLen = owl::length(edge.v1 - edge.v2);
      float v1Len = owl::length(edge.v1 - si.p);
      float v2Len = owl::length(edge.v2 - si.p);


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
