#pragma once

#include "owl/common/math/vec.h"
#include "owl/common/owl-common.h"
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

	int bvhIdx;
	int bvhHeight;

  int bspRoot;

  struct Span{
    // Spans of different compacted buffers
    // span.x -> start, span.y -> end;
    vec2i silSpan;
    vec2i edgeSpan;
    vec2i bspNodeSpan;
    vec2i triSpan;
    // vec2i octSpan[8]; // Spans for triangles lying different octants
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

struct LightBVH {
	vec3f aabbMin = vec3f(1e30f);
	vec3f aabbMax = vec3f(-1e30f);
	vec3f aabbMid = vec3f(0.f);
	// vec3f aabbMidEmit = vec3f(0.0f);
	float flux = 0.f;

	uint32_t left = 0, right = 0;
	uint32_t primIdx = 0, primCount = 0;
};

struct TriLight {
	vec3f aabbMin = vec3f(1e30f);
	vec3f aabbMax = vec3f(-1e30f);

	vec3f v1, v2, v3;
	vec3f cg;
	vec3f normal;
	vec3f emit;

	float flux;
	float area;
};

struct AABB { 
	vec3f bmin = vec3f(1e30f);
	vec3f bmax = vec3f(- 1e30f);

  __inline__ __both__
	void grow( vec3f p ) { bmin = owl::min( bmin, p ), bmax = owl::min( bmax, p ); }

  __inline__ __both__
  float area() 
	{ 
		vec3f e = bmax - bmin; // box extent
		return e.x * e.y + e.y * e.z + e.z * e.x; 
	}
};

struct LightEdge {
	// ids of adjecent faces and vertices are stored
	// we only support manifold meshes
	vec2i adjFaces;
	vec3f n1;
	vec3f n2;
	int adjFaceCount;
	vec3f v1;
	vec3f v2;
	vec3f cg1;
	vec3f cg2;
};
