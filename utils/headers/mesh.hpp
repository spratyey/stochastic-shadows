#pragma once

#include "owl/common/math/vec.h"
#include "types.hpp"
#include <vector>

using namespace owl;

/*! a simple indexed triangle mesh that our sample renderer will
    render */
class Mesh {
  public:
    std::vector<vec3f> vertex;
    std::vector<Edge>  edges;
    std::vector<Face>  faces;
    std::vector<vec3f> normal;

    std::vector<vec2f> texcoord;
    std::vector<vec3i> index;

    // material data:
    vec3f              diffuse;
    int                diffuseTextureID{ -1 };

    float              alpha;
    int                alphaTextureID{ -1 };

    vec3f emit;

    // Is light?
    bool isLight{ false };
    int lightIdx;

  void insertEdge(Edge &edge); 
  std::vector<int> getSilhouetteEdges(vec3f point);
  bool shouldFlip(Edge &edge, vec3f point);
};
