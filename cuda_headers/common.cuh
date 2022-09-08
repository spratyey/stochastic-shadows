#pragma once

#include "owl/owl.h"
#include "owl/common/math/vec.h"
#include "cuda_runtime.h"
#include "owl/common/math/random.h"
#include <optix_device.h>

using namespace owl;

#define PI 3.1415926f

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