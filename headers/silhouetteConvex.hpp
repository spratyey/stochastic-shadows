#pragma once

#include "types.hpp"
#include "bsp.hpp"
// #include "twoPassBsp.hpp"
#include "mesh.hpp"
#include "owl/common/math/vec.h"
#include <vector>

using namespace owl;

class ConvexSilhouette {
  public:
    MeshLight mesh;
    Mesh polyhedron; 

    std::vector<BSPNode> nodes;
    std::vector<int> silhouettes;
    std::vector<vec3f> vertices;
    int root;

	  ConvexSilhouette(Mesh &polyhedron); 
	  int makeLeaf(int index, BSP &bsp); 
    void GetNodeBuffer();
	  void GetSilhouetteBuffer();
    void GetVertexBuffer();
};
