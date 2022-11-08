#include "silhouetteConvex.hpp"

ConvexSilhouette::ConvexSilhouette(Mesh &polyhedron) : polyhedron(polyhedron) {
  std::vector<vec4f> planes;
  for (Face &face : polyhedron.faces) {
    planes.push_back(face.plane);
  }

  // TODO: Actually calculate bound or make it an param
	BSP bsp(planes, vec3f(-100), vec3f(100));

	root = bsp.root;
	
  for (auto node : bsp.nodes) {
    // TODO: Merge both the types
		nodes.push_back(BSPNode {
			node.plane,
			node.left,
			node.right
		});
	}

	// Create silhouette list and add dummy element to prevent index 0 from being used because
	// leaves are recognized by the sign of the index
	silhouettes.push_back(0);

	for (int i = 0; i < bsp.nodes.size(); i++) {
		BSPNode node = nodes[i];

		if (node.left < 0) {
			node.left = makeLeaf(-node.left, bsp);
		}

		if (node.right < 0) {
			node.right = makeLeaf(-node.right, bsp);
		}

		nodes[i] = node;
	}

  // TODO
	// this.vertices = polyhedron.vertices.Select(v => (Vector3)v).ToArray();
}

int ConvexSilhouette::makeLeaf(int index, BSP &bsp) {
  std::vector<int> silhouette = polyhedron.getSilhouetteEdges(bsp.leaves[index]);

  for (auto &sil : silhouette) {
    std::cerr << sil << " ";
  }
  std::cerr << std::endl;

	int left = silhouettes.size();
	silhouettes.insert(silhouettes.end(), silhouette.end(), silhouette.end());
	int right = silhouettes.size();
	
	nodes.push_back(BSPNode { left = -left, right = -right });

	return nodes.size() - 1;
}
