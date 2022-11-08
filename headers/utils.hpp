#pragma once

#include "owl/common/math/vec.h"
#include "utils.hpp"
#include "constants.cuh"

using namespace owl;

float getPlanePointDist(vec3f &point, vec4f &plane);
double divideSafe(double a, double b);
