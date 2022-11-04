#include "mesh.hpp"

void Mesh::insertEdge(Edge &edge) {
  Face f1 = faces[edge.adjFace1];
  Face f2 = faces[edge.adjFace2];

  // TODO: Fix this to work for boundary edges
  if (dot(cross(f1.n, f2.n), edge.vert2 - edge.vert1) < 0) {
    std::swap(edge.adjFace1, edge.adjFace2);
  }

  edges.push_back(edge);
}

std::vector<int> Mesh::getSilhouetteEdges(vec3f point) {
  std::vector<int> silEdges;
  for (int i = 0; i < edges.size(); i += 1) {
    Edge edge = edges[i];
    if (edge.numAdjFace == 0) continue;

    vec3f relative = edge.vert1 - point;

    bool isSil = false;
    Face f1 = faces[edge.adjFace1];
    bool visible1 = dot(relative, f1.n) > 0;
    if (edge.numAdjFace == 2) {
        Face f2 = faces[edge.adjFace2];
        isSil = dot(f1.n, relative) * dot(f2.n, relative) < 0;
    } else {
        isSil = dot(f1.n, relative) < 0;
    }


    if (isSil) {
      // Negetive index denotes need to flip
      silEdges.push_back(visible1 ? 1 : -1 * i);
    }
  }

  return silEdges;
}
