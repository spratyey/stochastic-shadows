#pragma once

#include "model.h"
#include "json.h"

struct Camera {
	vec3f from;
	vec3f at;
	vec3f up;
	float cosFovy;
};

struct Scene {
	// Scene contents
	Model* model;
	Model* triLights;
	std::vector<Camera> cameras;

	// Other information
	int spp;
	int imgWidth, imgHeight;
	std::string renderOutput;
	std::string renderStatsOutput;

	void syncLights();
};

bool parseScene(std::string sceneFile, Scene &scene);