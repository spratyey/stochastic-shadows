#pragma once

#include "owl/common/math/vec.h"
#include "bsp.hpp"
#include "utils.cuh"
#include <set>

using namespace owl;

class TwoPassBSP {
  private:
    std::vector<std::pair<int, vec3f>> points;
    std::vector<vec4f> planes;
    BSP *firstPass;
    int depth = 0;

  public:
    std::vector<vec3f> leaves;
    std::vector<BSPNode> nodes;
    int leafCount = 0;
    int averageDepth;
    int root;

    TwoPassBSP(std::vector<vec4f> &planes, vec3f minBound, vec3f maxBound);

    int makeNode(std::pair<int, int> &planeSpan, std::pair<int, int> &edgeSpan);
    int makeInnerNode(std::pair<int, int> &planeSpan, std::pair<int, int> &edgeSpan, vec4f &plane);
    int makeLeaf(std::pair<int, int> &pointSpan);
    std::pair<int, int> split(vec4f &plane, std::pair<int, int> &pointSpan);
    bool evaluateCut(vec4f &plane, std::pair<int, int> &pointSpan);
    void loadCube(vec3f min, vec3f max);

	  static double pseudoAngle(vec3f &up, vec3f &right, vec3f &v);
};
