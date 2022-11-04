#include "silhouetteConvex.hpp"

ConvexSilhouette::ConvexSilhouette() {
  std::vector<vec3f> planes = polyhedron.planes;
  // TODO: Actually calculate bound or make it an param
	BSP bsp(planes, vec3f(-1e6), vec3f(1e6));

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
			node.left = makeLeaf(-node.left, bsp, silhouettes, nodes);
		}

		if (node.right < 0) {
			node.right = makeLeaf(-node.right, bsp, silhouettes, nodes);
		}

		nodes[i] = node;
	}

  // TODO
	// this.vertices = polyhedron.vertices.Select(v => (Vector3)v).ToArray();
}

int ConvexSilhouette::makeLeaf(int index, BSP &bsp, std::vector<uint32_t> &silhouettes, std::vector<BSPNode> &nodes) {
	var silhouette = polyhedron.GetOrderedConvexSilhouette(bsp.leaves[index]);

	var left = silhouettes.Count;
	silhouettes.AddRange(silhouette);
	var right = silhouettes.Count;
	
	nodes.Add(new Node { left = -left, right = -right });

	return nodes.Count - 1;
}
