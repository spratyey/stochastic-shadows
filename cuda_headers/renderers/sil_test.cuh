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
      vec3f d = normalize(edge.v2 - edge.v1);
      vec3f closePoint = edge.v1 + d * clamp(dot(p - edge.v1, d), 0.f, length(edge.v2 - edge.v1));
      float perpDist = length(closePoint - p);
      if (perpDist < 0.1) {
        edgeIdx = j;
        lightIdx = i;
        if (shouldPrint) printf("%f\n", perpDist);

        if (shouldPrint) printf("p: %f %f %f\n", p.x, p.y, p.z);
        if (shouldPrint) printf("closestPoint: %f %f %f\n", closePoint.x, closePoint.y, closePoint.z);
        if (shouldPrint) printf("d: %f %f %f\n", d.x, d.y, d.z);
        if (shouldPrint) printf("p - edge.v1: %f %f %f\n", (p-edge.v1).x, (p-edge.v1).y, (p-edge.v1).z);
        if (shouldPrint) printf("edge.v1: %f %f %f\n", edge.v1.x, edge.v1.y, edge.v1.z);
        if (shouldPrint) printf("edge.v2: %f %f %f\n", edge.v2.x, edge.v2.y, edge.v2.z);
        if (shouldPrint) printf("%d %d\n", edgeIdx, lightIdx);
        break;
      }
    }
  }

  // if (shouldPrint) printf("%d %d\n", edgeIdx, lightIdx);

  if (lightIdx >= 0) {
    bool isSil = false;
    bool toFlip = false;
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
      isSil = dot(edge.n1, edge.v1 - camPos) * dot(edge.n2, edge.v2 - camPos) < 0;
    } else {
      isSil = true;
    }

    if (isSil) {
      bool toFlip = shouldFlip(edge, camPos);
    }
#endif

    if (isSil) {
      float edgeLen = length(edge.v1 - edge.v2);
      float v1Len = length(edge.v1 - p);
      float v2Len = length(edge.v2 - p);


      // Red -> v1, Green -> v2
      vec3f c1 = toFlip ? vec3f(1, 0, 0) : vec3f(0, 1, 0);
      vec3f c2 = vec3f(1, 1, 0) - c1;
      // if (shouldPrint) printf("%f %f %f\n", c1.x, c1.y, c1.z);
      // if (shouldPrint) printf("%f %f %f\n", c2.x, c2.y, c2.z);
      // if (shouldPrint) printf("%f %f %f\n", v1Len, v2Len, edgeLen);
      // if (shouldPrint) printf("%f %f %f\n", edge.v1.x, edge.v1.y, edge.v1.z);
      // if (shouldPrint) printf("%f %f %f\n", edge.v2.x, edge.v2.y, edge.v2.z);

      return v1Len / edgeLen * c1 + v2Len / edgeLen * c2;
    } else {
      return vec3f(0, 0, 1);
    }
  }

  return si.diffuse;
}
