#include "silhouetteConvex.hpp"

ConvexSilhouette::ConvexSilhouette(Mesh &polyhedron) : polyhedron(polyhedron) {
	std::vector<vec4f> planes;
	for (Face &face : polyhedron.faces) {
        planes.push_back(face.plane);
	}

	// TODO: Actually calculate bound or make it an param
	// BSP bsp(planes, vec3f(-100), vec3f(100));
	TwoPassBSP bsp(planes, vec3f(-10000), vec3f(10000));

	root = bsp.root;

    for (auto &node : bsp.nodes) {
		nodes.push_back(node);
	}

	// Create silhouette list and add dummy element to prevent index 0 from being used because
	// leaves are recognized by the sign of the index
	silhouettes.push_back(0);

    int leafCnt = 0;
	for (int i = 0; i < bsp.nodes.size(); i++) {
		BSPNode node = nodes[i];

		if (node.left < 0) {
			node.left = makeLeaf(-node.left, bsp);
            leafCnt += 1;
		}

		if (node.right < 0) {
			node.right = makeLeaf(-node.right, bsp);
            leafCnt += 1;
		}

		nodes[i] = node;
	}
}

int ConvexSilhouette::makeLeaf(int index, TwoPassBSP &bsp) {
    std::vector<int> silhouette = polyhedron.getSilhouetteEdges(bsp.leaves[index]);

	int left = silhouettes.size();
	silhouettes.insert(silhouettes.end(), silhouette.begin(), silhouette.end());
	int right = silhouettes.size();

	BSPNode node;
  	node.plane = vec4f(0);
  	node.silSpan = vec2i(left, right);
  	node.left = -1;
  	node.right = -1;

	nodes.push_back(node);

	return nodes.size() - 1;
}
