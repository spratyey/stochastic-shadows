#pragma once

#include "owl/owl.h"
#include "owl/common/math/vec.h"
#include "cuda_runtime.h"
#include "owl/common/math/random.h"
#include <optix_device.h>

using namespace owl;

#define PI 3.1415926f

struct SurfaceInteraction {
	bool hit;

	vec3f p;
	vec2f uv;
	vec3f wo, wi;
	vec3f wo_local, wi_local;

	vec3f n_geom, n_shad;

	vec3f diffuse;
	float alpha;

	vec3f emit;
	bool isLight;

	vec3f to_local[3], to_world[3];
};