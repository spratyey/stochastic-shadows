#pragma once

#include "common.cuh"

typedef owl::common::LCG<4> Random;

using namespace owl;

struct TriLight {
	vec3f v1, v2, v3;
	vec3f emissionRadiance;
};

struct LaunchParams {
	TriLight* areaLights;

	float4* accumBuffer;
	int accumId;
};

__constant__ LaunchParams optixLaunchParams;

struct RayGenData {
	uint32_t* frameBuffer;
	vec2i frameBufferSize;

	OptixTraversableHandle world;

	struct {
		vec3f pos;
		vec3f dir_00;
		vec3f dir_du;
		vec3f dir_dv;
	} camera;
};

struct TriangleMeshData {
	vec3f* vertex;
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