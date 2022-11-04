#include "mesh.hpp"

std::vector<int> Mesh::getSilhouetteEdges(vec3f point) {
  std::vector<int> silEdges;
  for (int i = 0; i < edges.size(); i += 1) {
    Edge edge = edges[i];
    if (edge.numAdjFace == 0) continue;

    bool isSil = false;
    Face f1 = faces[edge.adjFace1];
    if (edge.numAdjFace == 2) {
        Face f2 = faces[edge.adjFace2];
        isSil = dot(f1.n, edge.vert1 - point) * dot(f2.n, edge.vert1 - point) < 0;
    } else {
        isSil = dot(f1.n, edge.vert1 - point) < 0;
    }

    if (isSil) {
      silEdges.push_back(i);
    }
  }

  return silEdges;
}
