#include "bsp.hpp"

BSP::BSP(std::vector<vec4f> &planes, vec3f minBound, vec3f maxBound) {
  // Initialise random number generator
  std::random_device rd;
  gen = std::mt19937(rd());
  distr = std::uniform_real_distribution<>(0.5, 1.0);

  loadCube(minBound, maxBound);

  // TODO: Implement custom lambda to consider epsilon
  std::set<vec4f> planeSet;
  for (auto &plane : planes) {
    if (planeSet.find(plane) == planeSet.end()) {
      this->planes.push_back(plane);
      planeSet.insert(plane);
    }
  }
  std::cout << this->planes.size() << std::endl;

  leaves.push_back((minBound + maxBound) / 2);
  
  std::pair<int, int> planeSpan = std::make_pair(0, this->planes.size());
  std::pair<int, int> edgeSpan = std::make_pair(0, this->edges.size());
  root = makeNode(planeSpan, edgeSpan);
}

// Make a node and return its index in the node list. The index is positive for inner nodes and
// negative for leaves. A leaf is created if no plane subdivides the cell further.
int BSP::makeNode(std::pair<int, int> &planeSpan, std::pair<int, int> &edgeSpan) {
  int planeStart = planeSpan.first;
  int planeEnd = planeSpan.second;
  
  int planeIndex = -1;

  // Find valid planes which can cut the current cell
  for (int i = planeStart; i < planeEnd; i++) {
    vec4f plane = planes[i];
    bool isValid = testCut(plane, edgeSpan);

    if (isValid) {
      // NOTE: Order of below operations is important, don't switch
      planeIndex = planes.size();
      planes.push_back(plane);
    }
  }

  if (planeIndex == -1) {
    // If none, we are at leaf node
    return -makeLeaf(edgeSpan);
  } else {
    // Choose last valid plane to divide the current cell
    vec4f bestPlane = planes[planeIndex];
    planes.erase(planes.begin() + planeIndex);

    std::pair<int, int> newPlaneSpan = std::make_pair(planeEnd, planes.size());
    return makeInnerNode(newPlaneSpan, edgeSpan, bestPlane);
  }
}

// Creates an inner node that splits the current cell at the given plane and returns its index
int BSP::makeInnerNode(std::pair<int, int> &planeSpan, std::pair<int, int> &edgeSpan, vec4f &plane) {
  int planeStart = planeSpan.first;
  int planeEnd = planeSpan.second;

  int edgeStart = edgeSpan.first;

  BSPNode node  = {};
  node.plane = plane;
  node.silSpan = vec2i(-1, -1);
  node.left = -1;
  node.right = -1;

  int nodeIndex = nodes.size();
  nodes.push_back(node);

  std::pair<int, int> newEdgeSpan = split(plane, edgeSpan);
  node.left = makeNode(planeSpan, newEdgeSpan);
  vec4f negPlane(-plane.x, -plane.y, -plane.z, -plane.w);
  newEdgeSpan = split(negPlane, edgeSpan);
  node.right = makeNode(planeSpan, newEdgeSpan);

  // TODO: Verify this works
  edges.resize(edgeStart);
  planes.resize(planeStart);

  nodes[nodeIndex] = node;

  return nodeIndex;
}

// Creates a leaf and returns its index
int BSP::makeLeaf(std::pair<int, int> &edgeSpan) {
  int edgeStart = edgeSpan.first;
  int edgeEnd = edgeSpan.second;
  vec3f point = vec3f(0);
  float totalWeight = 0.0;

  for (int i = edgeStart; i < edgeEnd; i++) {
    std::pair<vec3f, vec3f> edge = edges[i];

    float weightA = distr(gen) / length(edge.first);
    float weightB = distr(gen) / length(edge.second);

    point += weightA * edge.first;
    point += weightB * edge.second;

    totalWeight += (weightA + weightB);
  }

  point /= totalWeight;

  leaves.push_back(point);

  return leaves.size() - 1;
}

// Test if plane would result in a valid cut with parts of the cell on either side
bool BSP::testCut(vec4f &plane, std::pair<int, int> &edgeSpan) {
  int edgesStart = edgeSpan.first;
  int edgesEnd = edgeSpan.second;

	int count1 = 0, count2 = 0;

	for (int i = edgesStart; i < edgesEnd; i++) {
		auto edge = edges[i];
		float distanceA = getPlanePointDist(edge.first, plane);
		float distanceB = getPlanePointDist(edge.second, plane);

		count1 += distanceA > +EPS ? 1 : 0;
		count2 += distanceA < -EPS ? 1 : 0;
		count1 += distanceB > +EPS ? 1 : 0;
		count2 += distanceB < -EPS ? 1 : 0;
	}

	return count1 > 0 && count2 > 0;
}

// Cuts away the parts of the cell on the negative side of the plane. This is done in two steps:
// 1. Cut each edge and discard it if it's entirely on the wrong side, keep track of
//    intersection points
// 2. Create new edges in the plane by adding the edges of the convex hull of the intersections.
//    This closes the cut faces of the convex polyhedron
// The resulting new cell is pushed onto the edge list and its index range is returned.
std::pair<int, int> BSP::split(vec4f &plane, std::pair<int, int> &edgeSpan) {
  int edgesStart = edgeSpan.first;
  int edgesEnd = edgeSpan.second;

  int newEdgesStart = edges.size();
  vec3f cutFaceCenter = vec3f(0);

  planeVertices.clear();

  // Split/discard existing edges
  for (int i = edgesStart; i < edgesEnd; i++) {
    std::pair<vec3f, vec3f> edge = edges[i];

    float distanceA = getPlanePointDist(edge.first, plane);
    float distanceB = getPlanePointDist(edge.second, plane);

    float t = divideSafe(distanceA, distanceA - distanceB);
    t = clamp(t, float(EPS), float(1 - EPS));
    vec3f intersection = edge.first + (edge.second - edge.first) * t;

    // Create new vertex at plane intersection
    if (distanceA < 0) edge.first = intersection;
    if (distanceB < 0) edge.second = intersection;

    float len = length(edge.first - edge.second);

    if (len > EPS) edges.push_back(edge);

    // Get the vertices that lie on the plane
    if ((distanceA < EPS && distanceB > -EPS) || (distanceA > -EPS && distanceB < EPS)) {
      planeVertices.push_back(std::make_pair(0, intersection));
      cutFaceCenter += intersection;
    }
  }

  if (planeVertices.size() < 3) {
    return std::make_pair(newEdgesStart, edges.size());
  }

  // Sort vertices in clip plane by angle around center
  cutFaceCenter /= planeVertices.size();

  vec3f up = normalize(planeVertices[0].second - cutFaceCenter);
  vec3f right = cross(normalize(vec3f(plane.x, plane.y, plane.z)), up);

  for (int i = 0; i < planeVertices.size(); i++) {
    vec3f vertVec = planeVertices[i].second - cutFaceCenter;
    planeVertices[i].first= pseudoAngle(up, right, vertVec);
  }

  std::sort(planeVertices.begin(), planeVertices.end(), [](auto &left, auto &right) {
    return left.first < right.first;
  });

  // Create new edges
  std::pair<vec3f, vec3f> newEdge;
  newEdge.first = planeVertices[planeVertices.size()- 1].second;

  for (int i = 0; i < planeVertices.size(); i++) {
    newEdge.second = planeVertices[i].second;
    edges.push_back(newEdge);
    newEdge.first = newEdge.second;
  }

  return std::make_pair(newEdgesStart, edges.size());
}

void BSP::loadCube(vec3f minBound, vec3f maxBound) {
  vec3f v1 = vec3f(minBound.x, minBound.y, minBound.z);
  vec3f v2 = vec3f(maxBound.x, minBound.y, minBound.z);
  vec3f v3 = vec3f(minBound.x, maxBound.y, minBound.z);
  vec3f v4 = vec3f(maxBound.x, maxBound.y, minBound.z);
  vec3f v5 = vec3f(minBound.x, minBound.y, maxBound.z);
  vec3f v6 = vec3f(maxBound.x, minBound.y, maxBound.z);
  vec3f v7 = vec3f(minBound.x, maxBound.y, maxBound.z);
  vec3f v8 = vec3f(maxBound.x, maxBound.y, maxBound.z);

  edges.push_back(std::make_pair(v1, v2));
  edges.push_back(std::make_pair(v3, v4));
  edges.push_back(std::make_pair(v5, v6));
  edges.push_back(std::make_pair(v7, v8));
  
  edges.push_back(std::make_pair(v1, v3));
  edges.push_back(std::make_pair(v2, v4));
  edges.push_back(std::make_pair(v5, v7));
  edges.push_back(std::make_pair(v6, v8));

  edges.push_back(std::make_pair(v1, v5));
  edges.push_back(std::make_pair(v2, v6));
  edges.push_back(std::make_pair(v3, v7));
  edges.push_back(std::make_pair(v4, v8));
}

// Sorting by this leads to the same order as sorting by angle
double BSP::pseudoAngle(vec3f &up, vec3f &right, vec3f &v) {
  float dx = dot(right, v);
  float dy = dot(up, v);

  if (std::abs(dx) > std::abs(dy)) {
    return (dx > 0 ? 0 : 4) + dy / dx;
  } else {
    return (dy > 0 ? 2 : 6) - dx / dy;
  }
}
