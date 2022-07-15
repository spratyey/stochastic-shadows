#pragma once

#include "utils.cuh"

struct SurfaceInteractionn {
	vec3f p;
	vec3f wo, wi;

	vec3f n_geom, n_shad;

	vec3f diffuse;
	float alpha;

	vec3f to_local[3], to_world[3];
};