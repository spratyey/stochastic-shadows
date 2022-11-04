#pragma once

#include "types.hpp"
#include "bsp.hpp"
#include "owl/common/math/vec.h"
#include <vector>

using namespace owl;

class ConvexSilhouette {
  public:
    MeshLight mesh;

    std::vector<BSPNode> nodes;
    std::vector<uint32_t> silhouettes;
    std::vector<vec3f> vertices;
    int root;

	  ConvexSilhouette(); 
	  int makeLeaf(int index, BSP &bsp, std::vector<uint32_t> &silhouettes, std::vector<BSPNode> &nodes); 
    void GetNodeBuffer();
	  void GetSilhouetteBuffer();
    void GetVertexBuffer();
};
