#include "owl/owl.h"
#include "owlViewer/OWLViewer.h"

#include "interactive.cuh"

#include "scene.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl2.h"

void drawUI();

using namespace owl;

// Compiled PTX code
extern "C" char interactive_ptx[];

struct RenderWindow : public owl::viewer::OWLViewer
{
    RenderWindow(Scene& scene);

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
     
    bool sbtDirty = true;

    OWLBuffer accumBuffer{ 0 };
    int accumId = 0;
     
    OWLRayGen rayGen{ 0 };
    OWLMissProg missProg{ 0 };
    
    OWLGroup world; // TLAS

    OWLContext context{ 0 };
    OWLModule module{ 0 };

    OWLParams launchParams; 
};

RenderWindow::RenderWindow(Scene& scene)
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
    module = owlModuleCreate(context, interactive_ptx);

    accumBuffer = owlDeviceBufferCreate(context, OWL_FLOAT4, 1, nullptr);
    owlBufferResize(accumBuffer, this->getWindowSize().x*this->getWindowSize().y);

    // ====================================================
    // Area lights setup (Assume triangular area lights)
    // ====================================================
    Model* triLights = scene.triLights;
    std::vector<TriLight> triLightList;

    for (auto light : triLights->meshes) {
        // Loop over index list (to setup individual triangles)
        // NOTE: all lights must be exported as triangular meshes
        // NOTE 2: the emission of the light is taken from its diffuse color component (see model.h)
        for (auto index : light->index) {
            TriLight lightData;

            lightData.v1 = light->vertex[index.x];
            lightData.v2 = light->vertex[index.y];
            lightData.v3 = light->vertex[index.z];

            lightData.emissionRadiance = light->emit;

            triLightList.push_back(lightData);
        }
    }

    // ====================================================
    // Launch Parameters setup
    // ====================================================
    OWLVarDecl launchParamsDecl[] = {
        {"triLights", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, areaLights)},
        {"accumBuffer", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, accumBuffer)},
        {"accumId", OWL_INT, OWL_OFFSETOF(LaunchParams, accumId)},
        {nullptr}
    };

    this->launchParams = owlParamsCreate(context, sizeof(LaunchParams), launchParamsDecl, -1);
    OWLBuffer triLightsBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(TriLight), triLightList.size(), triLightList.data());
    owlParamsSetBuffer(this->launchParams, "triLights", triLightsBuffer);
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

        // Create the actual geometry on the device
        OWLGeom triangleGeom = owlGeomCreate(context, triangleGeomType);

        // ====================================================
        // Data setup
        // ====================================================

        // Create CUDA buffers from mesh vertices, indices and UV coordinates
        OWLBuffer vertexBuffer = owlDeviceBufferCreate(context, OWL_FLOAT3, mesh->vertex.size(), mesh->vertex.data());
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
    owlMissProgSet3f(missProg, "const_color", owl3f{0.f, 0.f, 0.f});

    // ====================================================
    // Setup a pin-hole camera ray-gen program
    // ====================================================
    OWLVarDecl rayGenVars[] = {
        {"frameBuffer", OWL_RAW_POINTER, OWL_OFFSETOF(RayGenData, frameBuffer)},
        {"frameBufferSize", OWL_INT2, OWL_OFFSETOF(RayGenData, frameBufferSize)},
        {"world", OWL_GROUP, OWL_OFFSETOF(RayGenData, world)},
        {"camera.pos", OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.pos)},
        {"camera.dir_00", OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.dir_00)},
        {"camera.dir_du", OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.dir_du)},
        {"camera.dir_dv", OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.dir_dv)},
        {nullptr}
    };

    rayGen = owlRayGenCreate(context, module, "rayGen", sizeof(RayGenData), rayGenVars, -1);
    // Set the TLAS to be used in RayGen
    owlRayGenSetGroup(rayGen, "world", world);

    // ====================================================
    // Finally, build the programs, pipeline and SBT
    // ====================================================
    owlBuildPrograms(context);
    owlBuildPipeline(context);
    owlBuildSBT(context);
}

void RenderWindow::render()
{
    if (sbtDirty) {
        owlBuildSBT(context);
        sbtDirty = false;
    }

    owlParamsSet1i(this->launchParams, "accumId", this->accumId);

    owlLaunch2D(rayGen, this->fbSize.x, this->fbSize.y, this->launchParams);
    drawUI();

    accumId++;
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
    owlRayGenSet3f(rayGen, "camera.pos", (const owl3f&) camera_pos);
    owlRayGenSet3f(rayGen, "camera.dir_00", (const owl3f&) camera_d00);
    owlRayGenSet3f(rayGen, "camera.dir_du", (const owl3f&) camera_ddu);
    owlRayGenSet3f(rayGen, "camera.dir_dv", (const owl3f&) camera_ddv);
        
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

        ImGui::End();
    }

    ImGui::Render();
    ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
}

int main(int argc, char** argv)
{
    std::string currentScene;
    std::string defaultScene = "C:/Users/Projects/OptixRenderer/scenes/scene_configs/test_scene.json";

    if (argc == 1)
        currentScene = defaultScene;
    else
        currentScene = std::string(argv[1]);
    
    LOG("Loading scene " + currentScene);

    Scene scene;
    bool success = parseScene(currentScene, scene);
    if (!success) {
        LOG("Error loading scene");
        return -1;
    }

    RenderWindow win(scene);

    win.camera.setOrientation(scene.cameras[0].from,
        scene.cameras[0].at,
        scene.cameras[0].up,
        owl::viewer::toDegrees(acosf(scene.cameras[0].cosFovy)));
    win.enableFlyMode();
    win.enableInspectMode(owl::box3f(vec3f(-1000.f), vec3f(+1000.f)));
    win.setWorldScale(length(scene.model->bounds.span()));

    // ##################################################################
    // now that everything is ready: launch it ....
    // ##################################################################
    win.showAndRun();

    return 0;
}