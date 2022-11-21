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

  int edgeIdx = -1;
  int lightIdx = -1;
  for (int i = 0; i < optixLaunchParams.numMeshLights; i += 1) {
    int edgeStartIdx = optixLaunchParams.meshLights[i].spans.edgeSpan.x;
    int edgeEndIdx = optixLaunchParams.meshLights[i].spans.edgeSpan.y;
    for (int j = edgeStartIdx; j < edgeEndIdx; j += 1) {
      LightEdge edge = optixLaunchParams.lightEdges[j];
      vec3f d = normalize(edge.v2 - edge.v1);
      vec3f closePoint = edge.v1 + d * owl::clamp(dot(p - edge.v1, d), 0.f, length(edge.v2 - edge.v1));
      float perpDist = length(closePoint - p);
      if (perpDist < 0.1) {
        edgeIdx = j;
        lightIdx = i;
        break;
      }
    }
  }

  int silNum = 1;
  int silCount = 1;
  if (lightIdx >= 0) {
    bool isSil = false;
    bool toFlip = false;
    LightEdge edge = optixLaunchParams.lightEdges[edgeIdx];
#ifdef BSP_SIL
    BSPNode node = getSilEdges(lightIdx, camPos);
    MeshLight light = optixLaunchParams.meshLights[lightIdx];
    vec2i silSpan = node.silSpan;
    int edgeCount = light.spans.edgeSpan.y - light.spans.edgeSpan.x;
    int edgeStartIdx = light.spans.edgeSpan.x;
    int *silhouettes = &optixLaunchParams.silhouettes[light.spans.silSpan.x];
    toFlip = false;
    for (int i = silSpan.x; i < silSpan.y; i += 1) {
      int silIdx = silhouettes[i];
      if (abs(silIdx) == edgeIdx - edgeStartIdx || silIdx - edgeCount == edgeIdx - edgeStartIdx) {
        toFlip = shouldFlip(silIdx, edgeCount);
        isSil = true;
        silNum = i - silSpan.x + 1;
        silCount = silSpan.y - silSpan.x;
        if (shouldPrint) printf("%d ", silNum);
        break;
      }
    }
    if (shouldPrint) printf("\n");
#else
    if (edge.adjFaceCount == 2) {
      isSil = dot(edge.n1, edge.v1 - camPos) * dot(edge.n2, edge.v2 - camPos) < 0;
    } else {
      isSil = true;
    }

    if (isSil) {
      toFlip = shouldFlip(edge, camPos);
    }
#endif

    if (isSil) {
      float edgeLen = length(edge.v1 - edge.v2);
      float v1Len = length(edge.v1 - p);
      float v2Len = length(edge.v2 - p);

      // Red -> v1, Green -> v2
      vec3f c1 = toFlip ? vec3f(1, 0, 0) : vec3f(0, 1, 0);
      vec3f c2 = vec3f(1, 1, 0) - c1;

      return (v1Len / edgeLen * c1 + v2Len / edgeLen * c2) * ((float)silNum / (float)silCount);
    } else {
      return vec3f(0, 0, 1);
    }
  }

  return si.diffuse;
}
