#pragma once

using namespace owl;

enum RendererType {
	DIFFUSE=0,
	ALPHA=1,
	NORMALS=2,
	DIRECT_LIGHT=3,
	DIRECT_LIGHT_LBVH=4,
	LTC_BASELINE=5,
	NUM_RENDERER_TYPES
};

const char* rendererNames[NUM_RENDERER_TYPES] = {"Diffuse", "Alpha", "Normals", "Direct Light", 
												"Direct Light (Light BVH)", "LTC Baseline"};

#define RADIANCE_RAY_TYPE 0
#define SHADOW_RAY_TYPE 1

#ifdef __CUDA_ARCH__
typedef RayT<0, 2> RadianceRay;
typedef RayT<1, 2> ShadowRay;
#endif

struct TriLight {
	vec3f v1, v2, v3;
	vec3f cg;
	vec3f normal;
	vec3f emissionRadiance;
	float area;
};

struct LightBVH {
	vec3f aabbMin = vec3f(0.f);
	vec3f aabbMax = vec3f(0.f);
	vec3f aabbMid = vec3f(0.f);

	uint32_t left = 0, right = 0;
	uint32_t primIdx = 0, primCount = 0;
};

struct LaunchParams {
	TriLight* areaLights;
	int numAreaLights;

	LightBVH* areaLightsBVH;
	int areaLightsBVHHeight;

	float4* accumBuffer;
	int accumId;

	int rendererType;

	OptixTraversableHandle world;

	cudaTextureObject_t ltc_1, ltc_2, ltc_3;

	struct {
		vec3f pos;
		vec3f dir_00;
		vec3f dir_du;
		vec3f dir_dv;
	} camera;
};

__constant__ LaunchParams optixLaunchParams;

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
	vec3f visibility;
	vec3f point, normal;
	vec3f emit;
	float area;
};