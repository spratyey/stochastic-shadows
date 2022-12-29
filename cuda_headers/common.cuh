#pragma once

#include "owl/owl.h"
#include "owl/common/math/vec.h"
#include "cuda_runtime.h"
#include "owl/common/math/random.h"
#include "constants.cuh"
#include "types.hpp"

// Useful for debugging
// TODO: Override this if build is not debug mode
#define print_pixel(...)                                                   					\
{																							\
	const vec2i pixelId = owl::getLaunchIndex();											\
	if (optixLaunchParams.pixelId.x == pixelId.x && 										\
		optixLaunchParams.pixelId.y == optixLaunchParams.bufferSize.y - pixelId.y && 		\
		optixLaunchParams.clicked) 															\
		printf( __VA_ARGS__ );                                                              \
}

#define print_pixel_exact(pixelX, pixelY, ...)												\
{																							\
	const vec2i pixelId = owl::getLaunchIndex();											\
	if (pixelX == pixelId.x && pixelY == pixelId.y && 										\
		(optixLaunchParams.clicked || !optixLaunchParams.interactive))						\
	{ 																						\
		printf( __VA_ARGS__ );																\
	}																						\
}

using namespace owl;

#define RADIANCE_RAY_TYPE 0
#define SHADOW_RAY_TYPE 1

#ifdef __CUDA_ARCH__
typedef RayT<0, 2> RadianceRay;
typedef RayT<1, 2> ShadowRay;
#endif

struct LaunchParams {
	bool clicked;
	bool interactive;
	vec2i pixelId;

 	// Framebuffer
	int accumId;
	vec2i bufferSize;
#ifdef ACCUM
	float4* accumBuffer;
#endif
#ifdef SPATIAL_REUSE
	float3* normalBuffer;
	float3* albedoBuffer;
	float* depthBuffer;
	int *binIdxBuffer;
#endif
#if defined(USE_RESERVOIRS) && defined(SPATIAL_REUSE)
 // TODO: Implement sample count
	float4 *resFloatBuffer;
	int2 *resIntBuffer;
	uint passId;
#endif

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

#ifdef BSP_SIL
	int *silhouettes;
	BSPNode *bsp;
#endif

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