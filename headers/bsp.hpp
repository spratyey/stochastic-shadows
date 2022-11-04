#pragma once

#include "owl/common/math/vec.h"
#include "cuda_headers/constants.cuh"
#include "utils.hpp"
#include <random>
#include <vector>
#include "owl/common/math/vec/functors.h"

using namespace owl;

class BSP {
	struct Node {
		vec4f plane;
		int left;
		int right;
	};

  private:
    std::vector<std::pair<vec3f, vec3f>> edges;
    std::vector<vec4f> planes;
    std::vector<vec3f> leaves;
    std::vector<std::pair<float, vec3f>> planeVertices;
    std::mt19937 gen;
    std::uniform_real_distribution<> distr;

  public:
    std::vector<Node> nodes;
    int root;

    BSP(std::vector<vec4f> &planes, vec3f minBound, vec3f maxBound);

    int makeNode(std::pair<int, int> planeSpan, std::pair<int, int> edgeSpan);
    int makeInnerNode(std::pair<int, int> planeSpan, std::pair<int, int> edgeSpan, vec4f plane);
    int makeLeaf(std::pair<int, int> edgeSpan);
    std::pair<int, int> split(vec4f plane, std::pair<int, int> edgeSpan);
    bool testCut(vec4f plane, std::pair<int, int> edgeSpan);
    void loadCube(vec3f min, vec3f max);

	  static double pseudoAngle(vec3f up, vec3f right, vec3f v);
};
