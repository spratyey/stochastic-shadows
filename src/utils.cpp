#include "utils.hpp"

float getPlanePointDist(vec3f point, vec4f plane) {
  float dist = point.x * plane.x + point.y + plane.y + point.z * plane.z + 1;
  vec3f abc = vec3f(plane.x, plane.y, plane.z);
  return dist / length(abc);
}

double divideSafe(double a, double b) {
	if (b < EPS && b > -EPS) {
		b = b < 0 ? -EPS: EPS;
	}

	return a / b;
}
