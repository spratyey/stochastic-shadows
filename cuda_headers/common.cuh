#pragma once

#include "owl/owl.h"
#include "owl/common/math/vec.h"
#include "cuda_runtime.h"
#include "owl/common/math/random.h"
#include "constants.cuh"
#include "types.hpp"

using namespace owl;

enum RendererType {
	DIFFUSE=0,
	ALPHA=1,
	NORMALS=2,
	SILHOUETTE,
	DIRECT_LIGHT_LSAMPLE,
	DIRECT_LIGHT_BRDFSAMPLE,
	DIRECT_LIGHT_MIS,
	DIRECT_LIGHT_LBVH_LSAMPLE,
	DIRECT_LIGHT_LBVH_BRDFSAMPLE,
	DIRECT_LIGHT_LBVH_MIS,
	LTC_BASELINE,
	LTC_LBVH_LINEAR,
	LTC_LBVH_BST,
	LTC_LBVH_SILHOUTTE,
	NUM_RENDERER_TYPES
};

__inline__ __host__
bool CHECK_IF_LTC(RendererType t)
{
	switch (t) {
		case LTC_BASELINE:
		case LTC_LBVH_LINEAR:
		case LTC_LBVH_BST:
			return true;
		default:
			return false;
	}
}

#define RADIANCE_RAY_TYPE 0
#define SHADOW_RAY_TYPE 1

#ifdef __CUDA_ARCH__
typedef RayT<0, 2> RadianceRay;
typedef RayT<1, 2> ShadowRay;
#endif

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

struct LightBVH {
	vec3f aabbMin = vec3f(1e30f);
	vec3f aabbMax = vec3f(-1e30f);
	vec3f aabbMid = vec3f(0.f);
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

struct LaunchParams {
	bool clicked;
	vec2i pixelId;

	float4* accumBuffer;
	int accumId;

	int rendererType;
	OptixTraversableHandle world;
	cudaTextureObject_t ltc_1, ltc_2, ltc_3;

	TriLight* triLights;
	int numTriLights;

	LightEdge* lightEdges;
	int numLightEdges;

	MeshLight* meshLights;
	int numMeshLights;

	LightBVH* lightBlas;
	LightBVH* lightTlas;
	int lightTlasHeight;

	int *silhouettes;
	BSPNode *bsp;

	struct {
		vec3f pos;
		vec3f dir_00;
		vec3f dir_du;
		vec3f dir_dv;
	} camera;

	float lerp;
};

#ifdef __CUDA_ARCH__
__constant__ LaunchParams optixLaunchParams;
#endif

struct RayGenData {
	uint32_t* frameBuffer;
	vec2i frameBufferSize;
};

struct TriangleMeshData {
	vec3f* vertex;
	vec3f* normal;
	vec3i* index;
	vec2f* texCoord;

	bool isLight;
	vec3f emit;

	vec3f diffuse;
	bool hasDiffuseTexture;
	cudaTextureObject_t diffuse_texture;

	float alpha;
	bool hasAlphaTexture;
	cudaTextureObject_t alpha_texture;
};

struct MissProgData {
	vec3f const_color;
};

struct ShadowRayData {
	vec3f visibility = vec3f(0.f);
	vec3f point = vec3f(0.f), normal = vec3f(0.f), cg = vec3f(0.f);
	vec3f emit = vec3f(0.f);
	float area = 0.f;
};

struct AABB { 
	vec3f bmin = vec3f(1e30f);
	vec3f bmax = vec3f(- 1e30f);

	__inline__ __device__ __host__
    void grow( vec3f p ) { bmin = owl::min( bmin, p ), bmax = owl::min( bmax, p ); }

	__inline__ __device__ __host__ float area() 
    { 
        vec3f e = bmax - bmin; // box extent
        return e.x * e.y + e.y * e.z + e.z * e.x; 
    }
};

struct SurfaceInteraction {
	bool hit = false;

	vec3f p = vec3f(0.f);
	vec2f uv = vec2f(0.f);
	vec3f wo = vec3f(0.f), wi = vec3f(0.f);
	vec3f wo_local = vec3f(0.f), wi_local = vec3f(0.f);

	vec3f n_geom = vec3f(0.f), n_shad = vec3f(0.f);

	vec3f diffuse = vec3f(0.f);
	float alpha = 0.f;

	vec3f emit = vec3f(0.f);
	bool isLight = false;

	vec3f to_local[3], to_world[3];
};
