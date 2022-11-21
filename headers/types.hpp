#pragma once

#include "owl/common/math/vec.h"
#include <vector>

using namespace owl;

struct BSPNode {
  vec4f plane;
  vec2i silSpan;
  int left;
  int right;
};

struct MeshLight {
	vec3f aabbMin = vec3f(1e30f);
	vec3f aabbMax = vec3f(-1e30f);
	vec3f cg;
  vec3f avgEmit;

	float flux;

	int triIdx;
  int triStartIdx;
	int triCount;

	int bvhIdx;
	int bvhHeight;

  int bspRoot;

  struct Span{
    // Spans of different compacted buffers
    // span.x -> start, span.y -> end; span.z -> size
    // TODO: Remove the Z component
    vec3i silSpan;
    vec3i edgeSpan;
    vec3i bspNodeSpan;
  } spans;
};

struct Edge {
  // ids of adjacent faces and vertices are stored
  // non-manifold faces not supported
  int numAdjFace; // 0,1,2

  // ids of faces
  // these ids are local
  int adjFace1;
  int adjFace2;

  // ids of vertices
  // these ids are global
  // TODO: Check if this is needed
  int adjVert1;
  int adjVert2;

  // locations of vertices
  vec3f vert1;
  vec3f vert2;
};

class Face {
  public:
    // Vertex positions
    vec3f vp1, vp2, vp3;
    // Center of geometry
    vec3f cg;
    // Normal
    vec3f n;
    // Perpendicular distance to origin
    float d;
    // Equation of plane
    vec4f plane;

  Face(vec3f n1, vec3f n2, vec3f n3, vec3f vp1, vec3f vp2, vec3f vp3)
    : vp1(vp1), vp2(vp2), vp3(vp3) {
    n = (n1 + n2 + n3) / 3.f;
    cg = (vp1 + vp2 + vp3) / 3.f;
    d = dot(n, vp1);
    plane = vec4f(n, -d);
    d = std::abs(d);
  }
};
