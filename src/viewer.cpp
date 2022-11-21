#include "viewer.hpp"

RenderWindow::RenderWindow(Scene& scene, vec2i resolution, bool interactive, char *ptx) 
    : owl::viewer::OWLViewer("LTC Many Lights", resolution, interactive, false)
{
    this->currentScene = scene;
    this->initialize(scene, ptx);
}

void RenderWindow::setRendererType(RendererType type)
{
    this->rendererType = type;
}

float RenderWindow::evaluateSAHForLightBVH(LightBVH& node, std::vector<TriLight>& primitives, int axis, float pos)
{
    AABB leftBox, rightBox;
    int leftCount = 0, rightCount = 0;

    for (uint32_t i = node.primIdx; i < node.primCount; i++) {
        TriLight& light = primitives[i];

        if (light.cg[axis] < pos) {
            leftCount++;
            leftBox.grow(light.v1);
            leftBox.grow(light.v2);
            leftBox.grow(light.v3);
        }
        else {
            rightCount++;
            rightBox.grow(light.v1);
            rightBox.grow(light.v2);
            rightBox.grow(light.v3);
        }
    }

    float cost = leftCount * leftBox.area() + rightCount * rightBox.area();
    return cost > 0.f ? cost : 1e30f;
}

int RenderWindow::getLightBVHHeight(uint32_t nodeIdx, std::vector<LightBVH>& bvh)
{
    LightBVH& node = bvh[nodeIdx];
    if (node.primCount != 0)
        return 0;

    int leftHeight = getLightBVHHeight(node.left, bvh);
    int rightHeight = getLightBVHHeight(node.right, bvh);

    return max(leftHeight, rightHeight) + 1;
}

template <typename T>
void RenderWindow::updateLightBVHNodeBounds(uint32_t nodeIdx, std::vector<LightBVH> &bvh, std::vector<T> &primitives)
{
    bvh[nodeIdx].aabbMax = vec3f(-1e30f);
    bvh[nodeIdx].aabbMin = vec3f(1e30f);

    for (uint32_t i = bvh[nodeIdx].primIdx; i < bvh[nodeIdx].primIdx + bvh[nodeIdx].primCount; i++) {
        T& prim = primitives[i];

        bvh[nodeIdx].aabbMin = owl::min(bvh[nodeIdx].aabbMin, prim.aabbMin);
        bvh[nodeIdx].aabbMax = owl::max(bvh[nodeIdx].aabbMax, prim.aabbMax);
        bvh[nodeIdx].flux += prim.flux;
    }

    bvh[nodeIdx].aabbMid = (bvh[nodeIdx].aabbMin + bvh[nodeIdx].aabbMax) * 0.5f;
    bvh[nodeIdx].flux /= bvh[nodeIdx].primCount;
}

template <typename T>
void RenderWindow::subdivideLightBVH(uint32_t nodeIdx, std::vector<LightBVH>& bvh, std::vector<T>& primitives)
{
    if (bvh[nodeIdx].primCount <= 2) {
        bvh[nodeIdx].flux = 0.f;
        for (int z = bvh[nodeIdx].primIdx; z < bvh[nodeIdx].primIdx + bvh[nodeIdx].primCount; z++) {
            bvh[nodeIdx].flux += primitives[z].flux;
        }

        bvh[nodeIdx].flux /= bvh[nodeIdx].primCount;
    
        return;
    }
    
    vec3f extent = bvh[nodeIdx].aabbMax - bvh[nodeIdx].aabbMin;

    int axis = 0;
    if (extent.y < extent.x) axis = 1;
    if (extent.z < extent[axis]) axis = 2;
    float splitPos = bvh[nodeIdx].aabbMin[axis] + extent[axis] * 0.5f;

    int i = bvh[nodeIdx].primIdx;
    int j = i + bvh[nodeIdx].primCount - 1;
    while (i <= j) {
        if (primitives[i].cg[axis] < splitPos) {
            i++;
        }
        else {
            T iItem = primitives[i];
            T jItem = primitives[j];

            primitives[i] = jItem;
            primitives[j] = iItem;

            j--;
        }
    }

    int leftCount = i - bvh[nodeIdx].primIdx;
    if (leftCount == 0 || leftCount == bvh[nodeIdx].primCount) {
        bvh[nodeIdx].flux = 0.f;
        for (int z = bvh[nodeIdx].primIdx; z < bvh[nodeIdx].primIdx + bvh[nodeIdx].primCount; z++) {
            bvh[nodeIdx].flux += primitives[z].flux;
        }

        bvh[nodeIdx].flux /= bvh[nodeIdx].primCount;

        return;
    }

    bvh[nodeIdx].left = bvh.size();
    LightBVH leftNode;
    leftNode.primCount = leftCount;
    leftNode.primIdx = bvh[nodeIdx].primIdx;
    bvh.push_back(leftNode);

    bvh[nodeIdx].right = bvh.size();
    LightBVH rightNode;
    rightNode.primCount = bvh[nodeIdx].primCount - leftCount;
    rightNode.primIdx = i;
    bvh.push_back(rightNode);

    bvh[nodeIdx].primCount = 0;

    this->updateLightBVHNodeBounds<T>(bvh[nodeIdx].left, bvh, primitives);
    this->updateLightBVHNodeBounds<T>(bvh[nodeIdx].right, bvh, primitives);

    this->subdivideLightBVH<T>(bvh[nodeIdx].left, bvh, primitives);
    this->subdivideLightBVH<T>(bvh[nodeIdx].right, bvh, primitives);

    bvh[nodeIdx].flux = (bvh[bvh[nodeIdx].left].flux + bvh[bvh[nodeIdx].right].flux) / 2.0f;
}

void RenderWindow::initialize(Scene& scene, char *ptx)
{
    // Initialize IMGUI
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(this->handle, true);
    ImGui_ImplOpenGL2_Init();

    // Context & Module creation, accumulation buffer initialize
    context = owlContextCreate(nullptr, 1);
    module = owlModuleCreate(context, ptx);

    owlContextSetRayTypeCount(context, 2);

    // ====================================================
    // Area lights setup (Assume triangular area lights)
    // ====================================================
    LOG("Building individual light mesh BVH (BLAS) ...");

    Model* triLights = scene.triLights;

    int totalTri = 0;
#ifdef BSP_SIL
    std::vector<BSPNode> bspNodes;
    std::vector<int> silhouettes;
#endif
    for (auto light : triLights->meshes) {
        MeshLight meshLight;
        meshLight.flux = 0.f;
        meshLight.triIdx = this->triLightList.size();
        meshLight.triStartIdx = totalTri;
        meshLight.spans.edgeSpan = vec3i(lightEdgeList.size());

        // Calculate silhouette BSP 
#ifdef BSP_SIL
        ConvexSilhouette silhouette(*light);
        meshLight.spans.silSpan = vec3i(silhouettes.size());
        meshLight.spans.bspNodeSpan = vec3i(bspNodes.size());
        meshLight.bspRoot = silhouette.root;
        silhouettes.insert(silhouettes.end(), silhouette.silhouettes.begin(), silhouette.silhouettes.end());
        bspNodes.insert(bspNodes.end(), silhouette.nodes.begin(), silhouette.nodes.end());
#endif

        int numTri = 0;
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
            triLight.flux = 3.1415926f * triLight.area * (triLight.emit.x+triLight.emit.y+triLight.emit.z) / 3.f;

            triLight.aabbMin = owl::min(triLight.aabbMin, triLight.v1);
            triLight.aabbMin = owl::min(triLight.aabbMin, triLight.v2);
            triLight.aabbMin = owl::min(triLight.aabbMin, triLight.v3);

            triLight.aabbMax = owl::max(triLight.aabbMax, triLight.v1);
            triLight.aabbMax = owl::max(triLight.aabbMax, triLight.v2);
            triLight.aabbMax = owl::max(triLight.aabbMax, triLight.v3);

            triLightList.push_back(triLight); // append to a global list of all triangle light sources
            
            // Next, update the AABB and flux of current light mesh
            meshLight.aabbMin = owl::min(meshLight.aabbMin, triLight.aabbMin);
            meshLight.aabbMax = owl::max(meshLight.aabbMax, triLight.aabbMax);
            meshLight.flux += triLight.flux;

            // Keep track of number of triangles in the current light mesh
            numTri++;
        }

        // TODO: Move to a common edge representation similar to Face
        for (auto edge : light->edges) {
            LightEdge lightEdge;
            lightEdge.adjFaces.x = edge.adjFace1;
            lightEdge.n1 = triLightList[totalTri + lightEdge.adjFaces.x].normal;
            lightEdge.cg1 = triLightList[totalTri + lightEdge.adjFaces.x].cg;
            if (edge.numAdjFace == 2) {
                lightEdge.adjFaces.y = edge.adjFace2;
                lightEdge.n2 = triLightList[totalTri + lightEdge.adjFaces.y].normal;
                lightEdge.cg2 = triLightList[totalTri + lightEdge.adjFaces.y].cg;
            } else {
                lightEdge.adjFaces.y = -1;
            }

            lightEdge.v1 = edge.vert1;
            lightEdge.v2 = edge.vert2;
            lightEdge.adjFaceCount = edge.numAdjFace;

            lightEdgeList.push_back(lightEdge);
        }
                    
        totalTri += numTri;

        // Insert spans 
        meshLight.triCount = numTri;
        meshLight.spans.edgeSpan.y = lightEdgeList.size();
#ifdef BSP_SIL
        meshLight.spans.silSpan.y = silhouettes.size();
        meshLight.spans.bspNodeSpan.y = bspNodes.size();
        meshLight.spans.edgeSpan.z = meshLight.spans.edgeSpan.y - meshLight.spans.edgeSpan.x;
        meshLight.spans.silSpan.z = meshLight.spans.silSpan.y - meshLight.spans.silSpan.x;
        meshLight.spans.bspNodeSpan.z = meshLight.spans.bspNodeSpan.y - meshLight.spans.bspNodeSpan.x;
#endif

        meshLight.cg = (meshLight.aabbMin + meshLight.aabbMax) / 2.f;

        // Construct BVH for the current light mesh
        int rootNodeIdx = this->lightBlas.size(); // Root node index (BLAS since it consists of actual triangles)
        LightBVH root;
        root.left = root.right = 0;
        root.primIdx = meshLight.triIdx;
        root.primCount = meshLight.triCount;
        this->lightBlas.push_back(root);

        this->updateLightBVHNodeBounds<TriLight>(rootNodeIdx, this->lightBlas, this->triLightList);
        this->subdivideLightBVH<TriLight>(rootNodeIdx, this->lightBlas, this->triLightList);


        // Finally, set current light mesh parameters and addto a global list of all light meshes
        meshLight.bvhIdx = rootNodeIdx;
        meshLight.bvhHeight = this->getLightBVHHeight(rootNodeIdx, this->lightBlas);
        meshLightList.push_back(meshLight);
    }

    // Build the TLAS on light meshes (NOT on triangles)
    // Note, this is build on 'meshLightList', not on 'triLightList'
    LOG("Building BVH on meshes (TLAS) ...");

    LightBVH root;
    root.left = root.right = 0;
    root.primIdx = 0;
    root.primCount = meshLightList.size();
    this->lightTlas.push_back(root);

    this->updateLightBVHNodeBounds<MeshLight>(0, this->lightTlas, this->meshLightList);
    this->subdivideLightBVH<MeshLight>(0, this->lightTlas, this->meshLightList);
    this->lightTlasHeight = this->getLightBVHHeight(0, this->lightTlas);

    LOG("All light BVH built");

    // ====================================================
    // Launch Parameters setup
    // ====================================================
    OWLVarDecl launchParamsDecl[] = {
        // The actual light triangles
        {"triLights", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, triLights)},
        {"numTriLights", OWL_INT, OWL_OFFSETOF(LaunchParams, numTriLights)},
        // Light edges
        {"lightEdges", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, lightEdges)},
        {"numLightEdges", OWL_INT, OWL_OFFSETOF(LaunchParams, numLightEdges)},
        // The mesh lights
        {"meshLights", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, meshLights)},
        {"numMeshLights", OWL_INT, OWL_OFFSETOF(LaunchParams, numMeshLights)},
        // The light BLAS and TLAS
        {"lightBlas", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, lightBlas)},
        {"lightTlas", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, lightTlas)},
        {"lightTlasHeight", OWL_INT, OWL_OFFSETOF(LaunchParams, lightTlasHeight)},
        // BSP and silhouette
        {"silhouettes", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, silhouettes)},
        {"bsp", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, bsp)},
        {"world", OWL_GROUP, OWL_OFFSETOF(LaunchParams, world)},
        {"ltc_1", OWL_TEXTURE, OWL_OFFSETOF(LaunchParams, ltc_1)},
        {"ltc_2", OWL_TEXTURE, OWL_OFFSETOF(LaunchParams, ltc_2)},
        {"ltc_3", OWL_TEXTURE, OWL_OFFSETOF(LaunchParams, ltc_3)},
        {"camera.pos", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, camera.pos)},
        {"camera.dir_00", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, camera.dir_00)},
        {"camera.dir_du", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, camera.dir_du)},
        {"camera.dir_dv", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, camera.dir_dv)},
        {"rendererType", OWL_INT, OWL_OFFSETOF(LaunchParams, rendererType)},
        {"accumId", OWL_INT, OWL_OFFSETOF(LaunchParams, accumId)},
        {"accumBuffer", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, accumBuffer)},
        // Random controls
        {"lerp", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, lerp)},
        // Mouse variables
        {"clicked", OWL_BOOL, OWL_OFFSETOF(LaunchParams, clicked)},
        {"pixelId", OWL_INT2, OWL_OFFSETOF(LaunchParams, pixelId)},
        {nullptr}
    };

    this->launchParams = owlParamsCreate(context, sizeof(LaunchParams), launchParamsDecl, -1);

    // Random controls
    owlParamsSet1f(this->launchParams, "lerp", this->lerp);
    owlParamsSet1b(this->launchParams, "clicked", false);

    // Set LTC matrices (8x8, since only isotropic)
    OWLTexture ltc1 = owlTexture2DCreate(context, OWL_TEXEL_FORMAT_RGBA32F, 8, 8, ltc_iso_1,
                                            OWL_TEXTURE_LINEAR, OWL_TEXTURE_CLAMP);
    OWLTexture ltc2 = owlTexture2DCreate(context, OWL_TEXEL_FORMAT_RGBA32F, 8, 8, ltc_iso_2,
                                            OWL_TEXTURE_LINEAR, OWL_TEXTURE_CLAMP);
    OWLTexture ltc3 = owlTexture2DCreate(context, OWL_TEXEL_FORMAT_RGBA32F, 8, 8, ltc_iso_3,
                                            OWL_TEXTURE_LINEAR, OWL_TEXTURE_CLAMP);

    owlParamsSetTexture(this->launchParams, "ltc_1", ltc1);
    owlParamsSetTexture(this->launchParams, "ltc_2", ltc2);
    owlParamsSetTexture(this->launchParams, "ltc_3", ltc3);


    // Upload the <actual> triangle data for all area lights
    OWLBuffer triLightsBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(TriLight), triLightList.size(), triLightList.data());
    owlParamsSetBuffer(this->launchParams, "triLights", triLightsBuffer);
    owlParamsSet1i(this->launchParams, "numTriLights", this->triLightList.size());

    // Upload the <actual> light edge data for all area lights
    OWLBuffer lightEdgesBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(LightEdge), lightEdgeList.size(), lightEdgeList.data());
    owlParamsSetBuffer(this->launchParams, "lightEdges", lightEdgesBuffer);
    owlParamsSet1i(this->launchParams, "numLightEdges", lightEdgeList.size());

    // Upload the mesh data for all area lights
    OWLBuffer meshLightsBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(MeshLight), meshLightList.size(), meshLightList.data());
    owlParamsSetBuffer(this->launchParams, "meshLights", meshLightsBuffer);
    owlParamsSet1i(this->launchParams, "numMeshLights", this->meshLightList.size());

    // Upload the BLAS and TLAS for lights
    OWLBuffer lightBlasBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(LightBVH), lightBlas.size(), lightBlas.data());
    owlParamsSetBuffer(this->launchParams, "lightBlas", lightBlasBuffer);

    OWLBuffer lightTlasBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(LightBVH), lightTlas.size(), lightTlas.data());
    owlParamsSetBuffer(this->launchParams, "lightTlas", lightTlasBuffer);
    owlParamsSet1i(this->launchParams, "lightTlasHeight", lightTlasHeight);

#ifdef BSP_SIL
    // Upload the precomputed silhouettes
    OWLBuffer silBuffer = owlDeviceBufferCreate(context, OWL_INT, silhouettes.size(), silhouettes.data());
    owlParamsSetBuffer(this->launchParams, "silhouettes", silBuffer);

    // Upload BSP
    OWLBuffer bspBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(BSPNode), bspNodes.size(), bspNodes.data());
    owlParamsSetBuffer(this->launchParams, "bsp", bspBuffer);
#endif

    // ====================================================
    // Scene setup (scene geometry and materials)
    // ====================================================

    // Instance level accel. struct (IAS), built over geometry accel. struct (GAS) of each individual mesh
    std::vector<OWLGroup> blasList;

    // Loop over meshes, set up data and build a GAS on it. Add it to IAS.
    Model* model = scene.model;
    for (auto mesh : model->meshes) {

        // ====================================================
        // Initial setup 
        // ====================================================

        // TriangleMeshData is a CUDA struct. This declares variables to be set on the host (var names given as 1st entry)
        OWLVarDecl triangleGeomVars[] = {
            {"vertex", OWL_BUFPTR, OWL_OFFSETOF(TriangleMeshData, vertex)},
            {"normal", OWL_BUFPTR, OWL_OFFSETOF(TriangleMeshData, normal)},
            {"index", OWL_BUFPTR, OWL_OFFSETOF(TriangleMeshData, index)},
            {"texCoord", OWL_BUFPTR, OWL_OFFSETOF(TriangleMeshData, texCoord)},

            {"isLight", OWL_BOOL, OWL_OFFSETOF(TriangleMeshData, isLight)},
            {"emit", OWL_FLOAT3, OWL_OFFSETOF(TriangleMeshData, emit)},

            {"diffuse", OWL_FLOAT3, OWL_OFFSETOF(TriangleMeshData, diffuse)},
            {"diffuse_texture", OWL_TEXTURE, OWL_OFFSETOF(TriangleMeshData, diffuse_texture)},
            {"hasDiffuseTexture", OWL_BOOL, OWL_OFFSETOF(TriangleMeshData, hasDiffuseTexture)},

            {"alpha", OWL_FLOAT, OWL_OFFSETOF(TriangleMeshData, alpha)},
            {"alpha_texture", OWL_TEXTURE, OWL_OFFSETOF(TriangleMeshData, alpha_texture)},
            {"hasAlphaTexture", OWL_BOOL, OWL_OFFSETOF(TriangleMeshData, hasAlphaTexture)},

            {nullptr}
        };

        // This defines the geometry type of the variables defined above. 
        OWLGeomType triangleGeomType = owlGeomTypeCreate(context,
            /* Geometry type, in this case, a triangle mesh */
            OWL_GEOM_TRIANGLES,
            /* Size of CUDA struct */
            sizeof(TriangleMeshData),
            /* Binding to variables on the host */
            triangleGeomVars,
            /* num of variables, -1 implies sentinel is set */
            -1);

        // Defines the function name in .cu file, to be used for closest hit processing
        owlGeomTypeSetClosestHit(triangleGeomType, RADIANCE_RAY_TYPE, module, "triangleMeshCH");
        owlGeomTypeSetClosestHit(triangleGeomType, SHADOW_RAY_TYPE, module, "triangleMeshCHShadow");

        // Create the actual geometry on the device
        OWLGeom triangleGeom = owlGeomCreate(context, triangleGeomType);

        // ====================================================
        // Data setup
        // ====================================================

        // Create CUDA buffers from mesh vertices, indices and UV coordinates
        OWLBuffer vertexBuffer = owlDeviceBufferCreate(context, OWL_FLOAT3, mesh->vertex.size(), mesh->vertex.data());
        OWLBuffer normalBuffer = owlDeviceBufferCreate(context, OWL_FLOAT3, mesh->normal.size(), mesh->normal.data());
        OWLBuffer indexBuffer = owlDeviceBufferCreate(context, OWL_INT3, mesh->index.size(), mesh->index.data());
        OWLBuffer texCoordBuffer = owlDeviceBufferCreate(context, OWL_FLOAT2, mesh->texcoord.size(), mesh->texcoord.data());

        // Set emission value, and more importantly, if the current mesh is a light
        owlGeomSet1b(triangleGeom, "isLight", mesh->isLight);
        owlGeomSet3f(triangleGeom, "emit", owl3f{ mesh->emit.x, mesh->emit.y, mesh->emit.z });

        // Create CUDA buffers and upload them for diffuse and alpha textures
        if (mesh->diffuseTextureID != -1) {
            Texture* diffuseTexture = model->textures[mesh->diffuseTextureID];
            OWLTexture diffuseTextureBuffer = owlTexture2DCreate(context,
                OWL_TEXEL_FORMAT_RGBA8,
                diffuseTexture->resolution.x,
                diffuseTexture->resolution.y,
                diffuseTexture->pixel,
                OWL_TEXTURE_NEAREST,
                OWL_TEXTURE_CLAMP);
            owlGeomSetTexture(triangleGeom, "diffuse_texture", diffuseTextureBuffer);
            owlGeomSet1b(triangleGeom, "hasDiffuseTexture", true);
        }
        else {
            owlGeomSet3f(triangleGeom, "diffuse", owl3f{ mesh->diffuse.x, mesh->diffuse.y, mesh->diffuse.z });
            owlGeomSet1b(triangleGeom, "hasDiffuseTexture", false);
        }

        if (mesh->alphaTextureID != -1) {
            Texture* alphaTexture = model->textures[mesh->alphaTextureID];
            OWLTexture alphaTextureBuffer = owlTexture2DCreate(context,
                OWL_TEXEL_FORMAT_RGBA8,
                alphaTexture->resolution.x,
                alphaTexture->resolution.y,
                alphaTexture->pixel,
                OWL_TEXTURE_NEAREST,
                OWL_TEXTURE_CLAMP);
            owlGeomSetTexture(triangleGeom, "alpha_texture", alphaTextureBuffer);
            owlGeomSet1b(triangleGeom, "hasAlphaTexture", true);
        }
        else {
            owlGeomSet1f(triangleGeom, "alpha", mesh->alpha);
            owlGeomSet1b(triangleGeom, "hasAlphaTexture", false);
        }

        // ====================================================
        // Send the above data to device
        // ====================================================

        // Set vertices, indices and UV coords on the device
        owlTrianglesSetVertices(triangleGeom, vertexBuffer,
            mesh->vertex.size(), sizeof(vec3f), 0);
        owlTrianglesSetIndices(triangleGeom, indexBuffer,
            mesh->index.size(), sizeof(vec3i), 0);

        // TODO: Make a copy of these
        owlGeomSetBuffer(triangleGeom, "vertex", vertexBuffer);
        owlGeomSetBuffer(triangleGeom, "normal", normalBuffer);
        owlGeomSetBuffer(triangleGeom, "index", indexBuffer);
        owlGeomSetBuffer(triangleGeom, "texCoord", texCoordBuffer);

        // ====================================================
        // Build the BLAS (GAS)
        // ====================================================
        OWLGroup triangleGroup = owlTrianglesGeomGroupCreate(context, 1, &triangleGeom);
        owlGroupBuildAccel(triangleGroup);

        // Add to a list, which is later used to build the IAS
        blasList.push_back(triangleGroup);
    }

    // ====================================================
    // Build he TLAS (IAS)
    // ====================================================
    world = owlInstanceGroupCreate(context, blasList.size(), blasList.data());
    owlGroupBuildAccel(world);

    // ====================================================
    // Setup a generic miss program
    // ====================================================
    OWLVarDecl missProgVars[] = {
        {"const_color", OWL_FLOAT3, OWL_OFFSETOF(MissProgData, const_color)},
        {nullptr}
    };

    missProg = owlMissProgCreate(context, module, "miss", sizeof(MissProgData), missProgVars, -1);

    // Set a constant background color in the miss program (black for now)
    owlMissProgSet3f(missProg, "const_color", owl3f{ 0.f, 0.f, 0.f });

    // ====================================================
    // Setup a pin-hole camera ray-gen program
    // ====================================================
    OWLVarDecl rayGenVars[] = {
        {"frameBuffer", OWL_RAW_POINTER, OWL_OFFSETOF(RayGenData, frameBuffer)},
        {"frameBufferSize", OWL_INT2, OWL_OFFSETOF(RayGenData, frameBufferSize)},
        {nullptr}
    };

    rayGen = owlRayGenCreate(context, module, "rayGen", sizeof(RayGenData), rayGenVars, -1);
    // Set the TLAS to be used
    owlParamsSetGroup(this->launchParams, "world", world);

    // ====================================================
    // Finally, build the programs, pipeline and SBT
    // ====================================================
    owlBuildPrograms(context);
    owlBuildPipeline(context);
    owlBuildSBT(context);
}

void RenderWindow::mouseButtonLeft(const vec2i &where, bool pressed) {
    if (pressed == true) {
        owlParamsSet1b(this->launchParams, "clicked", true);
        owlParamsSet2i(this->launchParams, "pixelId", (const owl2i&)where);
    }
}

void RenderWindow::customKey(char key, const vec2i& pos)
{
    if (key == '1' || key == '!') {
        this->camera.setOrientation(this->camera.getFrom(), vec3f(0.f), vec3f(0.f, 0.f, 1.f), this->camera.getFovyInDegrees());
        this->cameraChanged();
    }
    else if (key == 'R' || key == 'r') {
        SceneCamera cam;
        cam.from = this->camera.getFrom();
        cam.at = this->camera.getAt();
        cam.up = this->camera.getUp();
        cam.cosFovy = this->camera.getCosFovy();

        this->recordedCameras.push_back(cam);
    }
    else if (key == 'F' || key == 'f') {
        nlohmann::json camerasJson;

        for (auto cam : this->recordedCameras) {
            nlohmann::json oneCameraJson;
            std::vector<float> from, at, up;

            for (int i = 0; i < 3; i++) {
                from.push_back(cam.from[i]);
                at.push_back(cam.at[i]);
                up.push_back(cam.up[i]);
            }

            oneCameraJson["from"] = from;
            oneCameraJson["to"] = at;
            oneCameraJson["up"] = up;
            oneCameraJson["cos_fovy"] = cam.cosFovy;

            camerasJson.push_back(oneCameraJson);
        }

        this->currentScene.json["cameras"] = camerasJson;
        std::ofstream outputFile(this->currentScene.jsonFilePath);
        outputFile << std::setw(4) << this->currentScene.json << std::endl;
    }
}

void RenderWindow::render()
{
    if (sbtDirty) {
        owlBuildSBT(context);
        sbtDirty = false;
    }

    owlLaunch2D(rayGen, this->fbSize.x, this->fbSize.y, this->launchParams);
    owlParamsSet1b(this->launchParams, "clicked", false);
}

void RenderWindow::resize(const vec2i& newSize)
{
    // Resize framebuffer, and other ops (OWL::Viewer ops)
    OWLViewer::resize(newSize);

    // Perform camera move i.e. set new camera parameters, and set SBT to be updated
    this->cameraChanged();
}

void RenderWindow::cameraChanged()
{
    const vec3f lookFrom = camera.getFrom();
    const vec3f lookAt = camera.getAt();
    const vec3f lookUp = camera.getUp();
    const float cosFovy = camera.getCosFovy();

    // ----------- compute variable values  ------------------
    vec3f camera_pos = lookFrom;
    vec3f camera_d00
        = normalize(lookAt - lookFrom);
    float aspect = fbSize.x / float(fbSize.y);
    vec3f camera_ddu
        = cosFovy * aspect * normalize(cross(camera_d00, lookUp));
    vec3f camera_ddv
        = cosFovy * normalize(cross(camera_ddu, camera_d00));
    camera_d00 -= 0.5f * camera_ddu;
    camera_d00 -= 0.5f * camera_ddv;

    // ----------- set variables  ----------------------------
    owlRayGenSet1ul(rayGen, "frameBuffer", (uint64_t) this->fbPointer);
    owlRayGenSet2i(rayGen, "frameBufferSize", (const owl2i&) this->fbSize);

    owlParamsSet3f(this->launchParams, "camera.pos", (const owl3f&)camera_pos);
    owlParamsSet3f(this->launchParams, "camera.dir_00", (const owl3f&) camera_d00);
    owlParamsSet3f(this->launchParams, "camera.dir_du", (const owl3f&) camera_ddu);
    owlParamsSet3f(this->launchParams, "camera.dir_dv", (const owl3f&) camera_ddv);
        
    this->sbtDirty = true;
}

void RenderWindow::drawUI()
{
    ImGui_ImplOpenGL2_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    {
        ImGui::Begin("Controls", NULL, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);
        ImGui::Text("%.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

        float currentLerp = this->lerp;
        ImGui::SliderFloat("LERP", &currentLerp, 0.f, 1.f);
        if (currentLerp != this->lerp) {
            this->lerp = currentLerp;
            owlParamsSet1f(this->launchParams, "lerp", this->lerp);
            this->cameraChanged();
        }

        ImGui::End();
    }

    ImGui::Render();
    ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
}