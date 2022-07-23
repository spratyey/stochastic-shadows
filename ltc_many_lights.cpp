#include "owl/owl.h"
#include "owl/common/math/random.h"
#include "owlViewer/OWLViewer.h"

#include "ltc_many_lights_cuda.cuh"

#include "scene.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl2.h"

#include "ltc_isotropic.h"

void drawUI();

using namespace owl;

// Compiled PTX code
extern "C" char ltc_many_lights_cuda_ptx[];

struct RenderWindow : public owl::viewer::OWLViewer
{
    RenderWindow(Scene& scene, vec2i resolution, bool interactive);

    void initialize(Scene& scene);

    /*! gets called whenever the viewer needs us to re-render out widget */
    void render() override;
    void drawUI() override;
    
    // /*! window notifies us that we got resized. We HAVE to override
    //     this to know our actual render dimensions, and get pointer
    //     to the device frame buffer that the viewer cated for us */
    void resize(const vec2i& newSize) override;
    
    // /*! this function gets called whenever any camera manipulator
    //   updates the camera. gets called AFTER all values have been updated */
    void cameraChanged() override;

    void customKey(char key, const vec2i& pos) override;
    void setRendererType(RendererType type);

    int getLightBVHHeight(uint32_t nodeIdx, std::vector<LightBVH>& bvh);
    float evaluateSAHForLightBVH(LightBVH& node, std::vector<TriLight>& primitives, int axis, float pos);
    void updateLightBVHNodeBounds(uint32_t nodeIdx, std::vector<LightBVH>& bvh, std::vector<TriLight>& primitives);
    void subdivideLightBVH(uint32_t nodeIdx, std::vector<LightBVH>& bvh, std::vector<TriLight>& primitives);
    
    RendererType rendererType;
    bool sbtDirty = true;

    OWLBuffer accumBuffer{ 0 };
    int accumId = 0;
     
    OWLRayGen rayGen{ 0 };
    OWLMissProg missProg{ 0 };
    
    OWLGroup world; // TLAS

    OWLContext context{ 0 };
    OWLModule module{ 0 };

    OWLParams launchParams;

    Scene currentScene;
    std::vector<SceneCamera> recordedCameras;

    std::vector<TriLight> triLightList;
    std::vector<LightBVH> lightBvh;
};

RenderWindow::RenderWindow(Scene& scene, vec2i resolution, bool interactive) 
    : owl::viewer::OWLViewer("LTC Many Lights", resolution, interactive, false)
{
    this->rendererType = DIRECT_LIGHT;
    this->currentScene = scene;
    this->initialize(scene);
}

void RenderWindow::setRendererType(RendererType type)
{
    this->rendererType = type;
    owlParamsSet1i(this->launchParams, "rendererType", (int)this->rendererType);
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

void RenderWindow::updateLightBVHNodeBounds(uint32_t nodeIdx, std::vector<LightBVH> &bvh, std::vector<TriLight> &primitives)
{
    LightBVH& node = bvh[nodeIdx];
    node.aabbMax = vec3f(-1e30f);
    node.aabbMin = vec3f(1e30f);

    for (uint32_t i = node.primIdx; i < node.primCount; i++) {
        TriLight& light = primitives[i];

        node.aabbMin = owl::min(node.aabbMin, light.v1);
        node.aabbMin = owl::min(node.aabbMin, light.v2);
        node.aabbMin = owl::min(node.aabbMin, light.v3);

        node.aabbMax = owl::max(node.aabbMax, light.v1);
        node.aabbMax = owl::max(node.aabbMax, light.v2);
        node.aabbMax = owl::max(node.aabbMax, light.v3);
    }

    node.aabbMid = (node.aabbMin + node.aabbMax) * 0.5f;
}

void RenderWindow::subdivideLightBVH(uint32_t nodeIdx, std::vector<LightBVH>& bvh, std::vector<TriLight>& primitives)
{
    if (bvh[nodeIdx].primCount <= 2) {
        bvh[nodeIdx].flux = 0.f;
        for (int z = bvh[nodeIdx].primIdx; z < bvh[nodeIdx].primCount; z++) {
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

    // int axis = -1;
    // float splitPos = 0.f, splitCost = 1e30f;
    // for (int a = 0; a < 3; a++) {
    //     for (uint32_t i = bvh[nodeIdx].primIdx; i < bvh[nodeIdx].primCount; i++) {
    //         TriLight& light = primitives[i];
    //         float candidatePos = light.cg[axis];
    //         float candidateCost = this->evaluateSAHForLightBVH(bvh[nodeIdx], primitives, a, candidatePos);
    //         if (candidateCost < splitCost) {
    //             splitCost = candidateCost;
    //             axis = a;
    //             splitPos = candidatePos;
    //         }
    //     }
    // }
    // 
    // vec3f e = bvh[nodeIdx].aabbMax - bvh[nodeIdx].aabbMin;
    // float parentArea = e.x * e.y + e.y * e.z + e.z * e.x;
    // float parentCost = bvh[nodeIdx].primCount * parentArea;
    // if (splitCost >= parentCost) {
    //     for (int z = bvh[nodeIdx].primIdx; z < bvh[nodeIdx].primCount; z++) {
    //         bvh[nodeIdx].flux += primitives[z].flux;
    //     }
    //     return;
    // }

    int i = bvh[nodeIdx].primIdx;
    int j = i + bvh[nodeIdx].primCount - 1;
    while (i <= j) {
        if (primitives[i].cg[axis] < splitPos) {
            i++;
        }
        else {
            TriLight iItem = primitives[i];
            TriLight jItem = primitives[j];

            primitives[i] = jItem;
            primitives[j] = iItem;

            j--;
        }
    }

    int leftCount = i - bvh[nodeIdx].primIdx;
    if (leftCount == 0 || leftCount == bvh[nodeIdx].primCount) {
        bvh[nodeIdx].flux = 0.f;
        for (int z = bvh[nodeIdx].primIdx; z < bvh[nodeIdx].primCount; z++) {
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

    this->updateLightBVHNodeBounds(bvh[nodeIdx].left, bvh, primitives);
    this->updateLightBVHNodeBounds(bvh[nodeIdx].right, bvh, primitives);

    this->subdivideLightBVH(bvh[nodeIdx].left, bvh, primitives);
    this->subdivideLightBVH(bvh[nodeIdx].right, bvh, primitives);

    bvh[nodeIdx].flux = (bvh[bvh[nodeIdx].left].flux + bvh[bvh[nodeIdx].right].flux) / 2.0;
}

void RenderWindow::initialize(Scene& scene)
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
    module = owlModuleCreate(context, ltc_many_lights_cuda_ptx);

    accumBuffer = owlDeviceBufferCreate(context, OWL_FLOAT4, 1, nullptr);
    owlBufferResize(accumBuffer, this->getWindowSize().x * this->getWindowSize().y);

    owlContextSetRayTypeCount(context, 2);

    // ====================================================
    // Area lights setup (Assume triangular area lights)
    // ====================================================
    Model* triLights = scene.triLights;

    for (auto light : triLights->meshes) {
        // Loop over index list (to setup individual triangles)
        // NOTE: all lights must be exported as triangular meshes
        // NOTE 2: the emission of the light is taken from its diffuse color component (see model.h)
        for (auto index : light->index) {
            TriLight lightData;

            lightData.v1 = light->vertex[index.x];
            lightData.v2 = light->vertex[index.y];
            lightData.v3 = light->vertex[index.z];
            lightData.cg = (lightData.v1 + lightData.v2 + lightData.v3) / 3.f;
            lightData.normal = normalize(light->normal[index.x]+light->normal[index.y]+light->normal[index.z]);
            lightData.area = 0.5f * length(cross(lightData.v1 - lightData.v2, lightData.v3 - lightData.v2));

            lightData.emissionRadiance = light->emit;
            lightData.flux = 3.1415926f * lightData.area * (light->emit.x+light->emit.y+light->emit.z) / 3.f;

            triLightList.push_back(lightData);
        }
    }

    // Build the Light BVH (ref: https://link.springer.com/chapter/10.1007/978-1-4842-4427-2_18)
    LOG("Building light BVH...");
    int numAreaLights = this->triLightList.size();

    LightBVH root;
    root.left = root.right = 0;
    root.primIdx = 0;
    root.primCount = numAreaLights;
    this->lightBvh.push_back(root);
    
    this->updateLightBVHNodeBounds(0, this->lightBvh, this->triLightList);
    this->subdivideLightBVH(0, this->lightBvh, this->triLightList);

    int lightBvhHeight = this->getLightBVHHeight(0, this->lightBvh);

    LOG(std::to_string(this->lightBvh[0].flux));

    LOG("Light BVH built");

    // ====================================================
    // Launch Parameters setup
    // ====================================================
    OWLVarDecl launchParamsDecl[] = {
        {"areaLights", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, areaLights)},
        {"numAreaLights", OWL_INT, OWL_OFFSETOF(LaunchParams, numAreaLights)},

        {"areaLightsBVH", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, areaLightsBVH)},
        {"areaLightsBVHHeight", OWL_INT, OWL_OFFSETOF(LaunchParams, areaLightsBVHHeight)},

        {"accumBuffer", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, accumBuffer)},
        {"accumId", OWL_INT, OWL_OFFSETOF(LaunchParams, accumId)},
        {"rendererType", OWL_INT, OWL_OFFSETOF(LaunchParams, rendererType)},
        {"world", OWL_GROUP, OWL_OFFSETOF(LaunchParams, world)},
        {"ltc_1", OWL_TEXTURE, OWL_OFFSETOF(LaunchParams, ltc_1)},
        {"ltc_2", OWL_TEXTURE, OWL_OFFSETOF(LaunchParams, ltc_2)},
        {"ltc_3", OWL_TEXTURE, OWL_OFFSETOF(LaunchParams, ltc_3)},
        {"camera.pos", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, camera.pos)},
        {"camera.dir_00", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, camera.dir_00)},
        {"camera.dir_du", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, camera.dir_du)},
        {"camera.dir_dv", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, camera.dir_dv)},
        {nullptr}
    };

    this->launchParams = owlParamsCreate(context, sizeof(LaunchParams), launchParamsDecl, -1);

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

    owlParamsSet1i(this->launchParams, "rendererType", (int)this->rendererType);

    // Upload the light BVH constructed above
    OWLBuffer lightBVHBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(LightBVH), this->lightBvh.size(), this->lightBvh.data());
    owlParamsSetBuffer(this->launchParams, "areaLightsBVH", lightBVHBuffer);
    owlParamsSet1i(this->launchParams, "areaLightsBVHHeight", lightBvhHeight);

    // Upload the <actual> data for all area lights
    OWLBuffer triLightsBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(TriLight), triLightList.size(), triLightList.data());
    owlParamsSetBuffer(this->launchParams, "areaLights", triLightsBuffer);
    owlParamsSet1i(this->launchParams, "numAreaLights", this->triLightList.size());

    owlParamsSet1i(this->launchParams, "accumId", this->accumId);
    owlParamsSetBuffer(this->launchParams, "accumBuffer", this->accumBuffer);

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
        owlGeomTypeSetClosestHit(triangleGeomType, 0, module, "triangleMeshCH");
        owlGeomTypeSetAnyHit(triangleGeomType, 1, module, "triangleMeshAH");

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

    if (CHECK_IF_LTC(this->rendererType) && accumId >= 1) {
        drawUI();
    }
    else {
        owlParamsSet1i(this->launchParams, "accumId", this->accumId);

        owlLaunch2D(rayGen, this->fbSize.x, this->fbSize.y, this->launchParams);
        accumId++;

        drawUI();
    }
}

void RenderWindow::resize(const vec2i& newSize)
{
    // Resize framebuffer, and other ops (OWL::Viewer ops)
    OWLViewer::resize(newSize);

    // Resize accumulation buffer, and set to launch params
    owlBufferResize(accumBuffer, newSize.x * newSize.y);
    owlParamsSetBuffer(this->launchParams, "accumBuffer", this->accumBuffer);

    // Perform camera move i.e. set new camera parameters, and set SBT to be updated
    this->cameraChanged();
}

void RenderWindow::cameraChanged()
{
    // Reset accumulation buffer, to restart MC sampling
    this->accumId = 0;

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

        int currentType = this->rendererType;
        ImGui::Combo("Renderer", &currentType, rendererNames, NUM_RENDERER_TYPES, 0);
        if (currentType != this->rendererType) {
            this->rendererType = static_cast<RendererType>(currentType);
            owlParamsSet1i(this->launchParams, "rendererType", currentType);
            this->cameraChanged();
        }

        ImGui::End();
    }

    ImGui::Render();
    ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
}

int main(int argc, char** argv)
{
    std::string savePath;
    bool isInteractive = true;

    std::string currentScene;
    std::string defaultScene = "C:/Users/Projects/OptixRenderer/scenes/scene_configs/test_scene.json";

    if (argc == 1)
        currentScene = defaultScene;
    else
        currentScene = std::string(argv[1]);

    if (argc >= 2) {
        isInteractive = atoi(argv[2]);
    }
    
    LOG("Loading scene " + currentScene);

    Scene scene;
    bool success = parseScene(currentScene, scene);
    if (!success) {
        LOG("Error loading scene");
        return -1;
    } 

    vec2i resolution(scene.imgWidth, scene.imgHeight);
    RenderWindow win(scene, resolution, isInteractive);

    if (isInteractive) {
        win.camera.setOrientation(scene.cameras[0].from,
            scene.cameras[0].at,
            scene.cameras[0].up,
            owl::viewer::toDegrees(acosf(scene.cameras[0].cosFovy)));
        win.enableFlyMode();
        win.enableInspectMode(owl::box3f(scene.model->bounds.lower, scene.model->bounds.upper));
        win.setWorldScale(length(scene.model->bounds.span()));

        // ##################################################################
        // now that everything is ready: launch it ....
        // ##################################################################
        win.showAndRun();
    }
    else {
        savePath = std::string(argv[3]);

        nlohmann::json stats;

        for (auto renderer : scene.renderers) {

            win.setRendererType(static_cast<RendererType>(renderer));
            std::string rendererName = rendererNames[renderer];
            
            int imgName = 0;    
            for (auto cam : scene.cameras) {
                win.camera.setOrientation(cam.from, cam.at, cam.up, owl::viewer::toDegrees(acosf(cam.cosFovy)));
                win.resize(resolution);

                auto start = std::chrono::high_resolution_clock::now();

                win.accumId = 0;
                for (int sample = 0; sample < scene.spp; sample++) {
                    win.render();
                }

                auto finish = std::chrono::high_resolution_clock::now();

                auto milliseconds_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count() / 1e6;

                std::string imgFileName = savePath + "/" + rendererName + "_" + std::to_string(imgName) + ".png";
                nlohmann::json currentStats = {
                    {"image_name", imgFileName},
                    {"spp", scene.spp},
                    {"width", scene.imgWidth},
                    {"height", scene.imgHeight},
                    {"frametime_milliseconds", milliseconds_taken},
                    {"num_area_lights", win.triLightList.size()},
                    {"renderer", rendererName}
                };

                stats.push_back(currentStats);

                win.screenShot(imgFileName);
                imgName++;
            }
        }

        std::ofstream op(savePath + "/stats.json");
        op << std::setw(4) << stats << std::endl;
    }

    return 0;
}