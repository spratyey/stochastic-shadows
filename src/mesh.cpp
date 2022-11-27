#include "mesh.hpp"

void Mesh::insertEdge(Edge &edge) {
  if (edge.numAdjFace == 0) return;

  Face f1 = faces[edge.adjFace1];
  // TODO: Fix this to work for boundary edges
  if (edge.numAdjFace == 2) {
    Face f2 = faces[edge.adjFace2];

    if (dot(cross(f1.n, f2.n), edge.vert2 - edge.vert1) < 0) {
      std::swap(edge.adjFace1, edge.adjFace2);
    }
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
      bool toFlip = !visible1;
      if (edge.numAdjFace == 1) {
        // Special case for boundary edges
        toFlip = dot(edge.vert2 - edge.vert1, cross(f1.cg - edge.vert1, f1.n)) > 0;
      }

      // Negative index denotes need to flip
      // For index 0, edges.size() denotes need to flip
      if (i == 0 && toFlip) {
        silEdges.push_back(edges.size());
      } else {
        silEdges.push_back((toFlip ? -1 : 1) * i);
      }
    }
  }

  if (silEdges.empty()) {
    return silEdges;
  }

  std::vector<int> orderedSilEdges;
  bool toFlip = false;
  if (silEdges[0] < 0 || silEdges[0] == edges.size()) {
    toFlip = true;
  }
  int edgeIdx = std::abs(silEdges[0]) % edges.size();
  int remaining  = silEdges.size();
  int curId = toFlip ? edges[edgeIdx].adjVert2 : edges[edgeIdx].adjVert1;
  while (remaining > 0) {
    bool found = false;
    for (auto silEdge : silEdges) {
      toFlip = false;
      if (silEdge < 0 || silEdge == edges.size()) {
        toFlip = true;
      }

      edgeIdx = std::abs(silEdge) % edges.size();
      if (toFlip) {
        if (curId == edges[edgeIdx].adjVert2) {
          curId = edges[edgeIdx].adjVert1;
          orderedSilEdges.push_back(silEdge);
          remaining -= 1;
          found = true;
          break;
        }
      } else {
        if (curId == edges[edgeIdx].adjVert1) {
          curId = edges[edgeIdx].adjVert2;
          orderedSilEdges.push_back(silEdge);
          remaining -= 1;
          found = true;
          break;
        }
      }
    }
    if (!found) break;
  }

  return orderedSilEdges;
}
