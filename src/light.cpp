#include "light.hpp"

void LightInfo::initialize(Scene &scene, bool calcSilhouette) {
    // ====================================================
    // Area lights setup (Assume triangular area lights)
    // ====================================================
    LOG("Building individual light mesh BVH (BLAS) ...");

    Model* triLights = scene.triLights;

    int totalTri = 0;

    for (auto light : triLights->meshes) {
        MeshLight meshLight;
        meshLight.flux = 0.f;
        meshLight.spans.triSpan = this->triLightList.size();
        meshLight.spans.edgeSpan = vec2i(this->lightEdgeList.size());

        // Calculate silhouette BSP 
        if (calcSilhouette) {
            ConvexSilhouette silhouette(*light);
            meshLight.spans.silSpan = vec2i(this->silhouettes.size());
            meshLight.spans.bspNodeSpan = vec2i(this->bspNodes.size());
            meshLight.bspRoot = silhouette.root;
            meshLight.avgEmit = vec3f(0);
            this->silhouettes.insert(this->silhouettes.end(), silhouette.silhouettes.begin(), silhouette.silhouettes.end());
            this->bspNodes.insert(this->bspNodes.end(), silhouette.nodes.begin(), silhouette.nodes.end());
        }

        int numTri = 0;
        float totalArea = 0;

        std::vector<TriLight> lightBins[NUM_BINS];
        for (auto index : light->index) {
            // First, setup data foran individual triangle light source
            TriLight triLight;

            triLight.v1 = light->vertex[index.x];
            triLight.v2 = light->vertex[index.y];
            triLight.v3 = light->vertex[index.z];

            triLight.cg = (triLight.v1 + triLight.v2 + triLight.v3) / 3.f;
            triLight.normal = normalize(light->normal[index.x] + light->normal[index.y] + light->normal[index.z]);
            triLight.area = 0.5f * length(cross(triLight.v1 - triLight.v2, triLight.v3 - triLight.v2));

            triLight.emit = light->emit;
            triLight.flux = triLight.area * length(triLight.emit);

            triLight.aabbMin = owl::min(triLight.aabbMin, triLight.v1);
            triLight.aabbMin = owl::min(triLight.aabbMin, triLight.v2);
            triLight.aabbMin = owl::min(triLight.aabbMin, triLight.v3);

            triLight.aabbMax = owl::max(triLight.aabbMax, triLight.v1);
            triLight.aabbMax = owl::max(triLight.aabbMax, triLight.v2);
            triLight.aabbMax = owl::max(triLight.aabbMax, triLight.v3);

            // Calculate in which octant does the normal lie
            int binIdx = rand() % NUM_BINS;
            lightBins[binIdx].push_back(triLight);
            
            // Next, update the AABB and flux of current light mesh
            meshLight.aabbMin = owl::min(meshLight.aabbMin, triLight.aabbMin);
            meshLight.aabbMax = owl::max(meshLight.aabbMax, triLight.aabbMax);
            meshLight.flux += triLight.flux;
            
            // Set average emmitance weighted by triangle size
            meshLight.avgEmit += triLight.area * light->emit;

            // Keep track of number of triangles in the current light mesh
            numTri++;

            // Keep track of total triangle area
            totalArea += triLight.area;
        }

        for (int i = 0; i < NUM_BINS; i += 1) {
            meshLight.spans.binSpan[i].x = this->triLightList.size();
            this->triLightList.insert(this->triLightList.end(), lightBins[i].begin(), lightBins[i].end());
            meshLight.spans.binSpan[i].y = this->triLightList.size();
        }

        meshLight.avgEmit /= totalArea;

        // TODO: Move to a common edge representation similar to Face
        for (auto edge : light->edges) {
            LightEdge lightEdge;
            lightEdge.adjFaces.x = edge.adjFace1;
            lightEdge.n1 = this->triLightList[totalTri + lightEdge.adjFaces.x].normal;
            lightEdge.cg1 = this->triLightList[totalTri + lightEdge.adjFaces.x].cg;
            if (edge.numAdjFace == 2) {
                lightEdge.adjFaces.y = edge.adjFace2;
                lightEdge.n2 = this->triLightList[totalTri + lightEdge.adjFaces.y].normal;
                lightEdge.cg2 = this->triLightList[totalTri + lightEdge.adjFaces.y].cg;
            } else {
                lightEdge.adjFaces.y = -1;
            }

            lightEdge.v1 = edge.vert1;
            lightEdge.v2 = edge.vert2;
            lightEdge.adjFaceCount = edge.numAdjFace;

           this->lightEdgeList.push_back(lightEdge);
        }
                    
        totalTri += numTri;

        // Insert spans 
        meshLight.spans.triSpan.y = this->triLightList.size();
        meshLight.spans.edgeSpan.y = this->lightEdgeList.size();

        if (calcSilhouette) {
            meshLight.spans.silSpan.y = this->silhouettes.size();
            meshLight.spans.bspNodeSpan.y = this->bspNodes.size();
        }

        meshLight.cg = (meshLight.aabbMin + meshLight.aabbMax) / 2.f;

        // Construct BVH for the current light mesh
        int rootNodeIdx = this->lightBlas.size(); // Root node index (BLAS since it consists of actual triangles)
        LightBVH root;
        root.left = root.right = 0;
        root.primIdx = meshLight.spans.triSpan.x;
        root.primCount = meshLight.spans.triSpan.y - meshLight.spans.triSpan.x;
        this->lightBlas.push_back(root);

        updateLightBVHNodeBounds<TriLight>(rootNodeIdx, this->lightBlas, this->triLightList);
        subdivideLightBVH<TriLight>(rootNodeIdx, this->lightBlas, this->triLightList);


        // Finally, set current light mesh parameters and addto a global list of all light meshes
        meshLight.bvhIdx = rootNodeIdx;
        meshLight.bvhHeight = getLightBVHHeight(rootNodeIdx, this->lightBlas);
        this->meshLightList.push_back(meshLight);
    }

    // Build the TLAS on light meshes (NOT on triangles)
    // Note, this is build on 'meshLightList', not on 'triLightList'
    LOG("Building BVH on meshes (TLAS) ...");

    LightBVH root;
    root.left = root.right = 0;
    root.primIdx = 0;
    root.primCount = meshLightList.size();
    this->lightTlas.push_back(root);

    updateLightBVHNodeBounds<MeshLight>(0, this->lightTlas, this->meshLightList);
    subdivideLightBVH<MeshLight>(0, this->lightTlas, this->meshLightList);
    this->lightTlasHeight = getLightBVHHeight(0, this->lightTlas);
    std::cout << "BVH Height: " << this->lightTlasHeight << std::endl;
    LOG("All light BVH built");
}

void LightInfo::write(std::ofstream &stream) {
    serialize_vector(triLightList, stream);
    serialize_vector(meshLightList, stream);
    serialize_vector(lightBlas, stream);
    serialize_vector(lightTlas, stream);
    serialize_vector(lightEdgeList, stream);
    serialize_vector(bspNodes, stream);
    serialize_vector(silhouettes, stream);
    stream.write((char *)&this->lightTlasHeight, sizeof(this->lightTlasHeight));
}

void LightInfo::read(std::ifstream &stream) {
    deserialize_vector(triLightList, stream);
    deserialize_vector(meshLightList, stream);
    deserialize_vector(lightBlas, stream);
    deserialize_vector(lightTlas, stream);
    deserialize_vector(lightEdgeList, stream);
    deserialize_vector(bspNodes, stream);
    deserialize_vector(silhouettes, stream);
    stream.read((char *)&this->lightTlasHeight, sizeof(this->lightTlasHeight));
    std::cout << this->lightTlasHeight << std::endl;
}