#pragma once

#define MAX_LTC_LIGHTS 20

using namespace owl;

enum RendererType {
	DIFFUSE=0,
	ALPHA=1,
	NORMALS=2,
	DIRECT_LIGHT_LSAMPLE,
	DIRECT_LIGHT_BRDFSAMPLE,
	DIRECT_LIGHT_MIS,
	DIRECT_LIGHT_LBVH_LSAMPLE,
	DIRECT_LIGHT_LBVH_BRDFSAMPLE,
	DIRECT_LIGHT_LBVH_MIS,
	LTC_BASELINE,
	LTC_LBVH_LINEAR,
	LTC_LBVH_BST,
	NUM_RENDERER_TYPES
};

const char* rendererNames[NUM_RENDERER_TYPES] = {"Diffuse", "Alpha", "Normals",
												"Direct Light (Light)", "Direct Light (BRDF)", "Direct Light (MIS)",
												"Direct Light (Light BVH) (Light)", "Direct Light (Light BVH) (BRDF)", "Direct Light (Light BVH) (MIS)",
												"LTC Baseline", "LTC (Light BVH, Linear)", "LTC (Light BVH, BST)" };

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

struct MeshLight {
	vec3f aabbMin = vec3f(1e30f);
	vec3f aabbMax = vec3f(-1e30f);
	vec3f cg;
	float flux;

	int triIdx;
	int triCount;

	int bvhIdx;
	int bvhHeight;
};

struct LaunchParams {
	float4* accumBuffer;
	int accumId;

	int rendererType;
	OptixTraversableHandle world;
	cudaTextureObject_t ltc_1, ltc_2, ltc_3;

	TriLight* triLights;
	int numTriLights;

	MeshLight* meshLights;
	int numMeshLights;

	LightBVH* lightBlas;
	LightBVH* lightTlas;
	int lightTlasHeight;

	struct {
		vec3f pos;
		vec3f dir_00;
		vec3f dir_du;
		vec3f dir_dv;
	} camera;

	float lerp;
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
	vec3f point, normal, cg;
	vec3f emit;
	float area;
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