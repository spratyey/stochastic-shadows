#include "viewer.hpp"

RenderWindow::RenderWindow(Scene& scene, vec2i resolution, bool interactive, char *ptx) 
    : owl::viewer::OWLViewer("LTC Many Lights", resolution, interactive, false)
{
    this->currentScene = scene;
    this->initialize(scene, ptx, interactive);
}


void RenderWindow::initialize(Scene& scene, char *ptx, bool interactive)
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

    accumBuffer = owlDeviceBufferCreate(context, OWL_FLOAT4, 1, nullptr);
    owlBufferResize(accumBuffer, this->getWindowSize().x * this->getWindowSize().y);

    owlContextSetRayTypeCount(context, 2);

    // Load light information
    LightInfo lightInfo;
    auto lightInformationPath = scene.json["light_information"];
    std::ifstream in_file(lightInformationPath, std::ios::binary);
    lightInfo.read(in_file);
    this->triLightList = lightInfo.triLightList;
    this->meshLightList = lightInfo.meshLightList;
    this->lightTlas = lightInfo.lightTlas;
    this->lightBlas = lightInfo.lightBlas;
    this->lightTlasHeight = lightInfo.lightTlasHeight;
    this->lightEdgeList = lightInfo.lightEdgeList;
    std::vector<BSPNode> bspNodes = lightInfo.bspNodes;
    std::vector<int> silhouettes = lightInfo.silhouettes;

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
        // Screen size
        {"bufferSize", OWL_INT2, OWL_OFFSETOF(LaunchParams, bufferSize)},
        {"interactive", OWL_BOOL, OWL_OFFSETOF(LaunchParams, interactive)},
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

    // Set screen size
    vec2i bufferSize(this->fbSize.x, this->fbSize.y);
    owlParamsSet2i(this->launchParams, "bufferSize", (const owl2i&)bufferSize);

    // Upload accumulation buffer and ID
    owlParamsSet1i(this->launchParams, "accumId", this->accumId);
    owlParamsSetBuffer(this->launchParams, "accumBuffer", this->accumBuffer);

#ifdef BSP_SIL
    // Upload the precomputed silhouettes
    OWLBuffer silBuffer = owlDeviceBufferCreate(context, OWL_INT, silhouettes.size(), silhouettes.data());
    owlParamsSetBuffer(this->launchParams, "silhouettes", silBuffer);

    // Upload BSP
    OWLBuffer bspBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(BSPNode), bspNodes.size(), bspNodes.data());
    owlParamsSetBuffer(this->launchParams, "bsp", bspBuffer);
#endif

    owlParamsSet1b(this->launchParams, "interactive", interactive);

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

#if RENDERER == DIRECT_LIGHTING
    owlParamsSet1i(this->launchParams, "accumId", this->accumId);

    owlLaunch2D(rayGen, this->fbSize.x, this->fbSize.y, this->launchParams);
    accumId++;
#endif

    OptixDenoiserParams denoiserParams;
#if OPTIX_VERSION > 70500
    denoiserParams.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_ALPHA_AS_AOV;
#endif
    denoiserParams.hdrIntensity = (CUdeviceptr)0;
    denoiserParams.blendFactor = 0.0f;

    // -------------------------------------------------------
    OptixImage2D inputLayer;
    inputLayer.data = (CUdeviceptr)this->fbPointer;
    /// Width of the image (in pixels)
    
    inputLayer.width = this->fbSize.x;;
    /// Height of the image (in pixels)
    inputLayer.height = this->fbSize.y;
    /// Stride between subsequent rows of the image (in bytes).
    inputLayer.rowStrideInBytes = this->fbSize.x * sizeof(float4);
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
    inputLayer.pixelStrideInBytes = sizeof(float4);
    /// Pixel format.
    inputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    // -------------------------------------------------------
    OptixImage2D outputLayer;
    outputLayer.data = (CUdeviceptr)denoisedBuffer.get();
    /// Width of the image (in pixels)
    outputLayer.width = this->fbSize.x;
    /// Height of the image (in pixels)
    outputLayer.height = this->fbSize.y;
    /// Stride between subsequent rows of the image (in bytes).
    outputLayer.rowStrideInBytes = this->fbSize.x * sizeof(float4);
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
    outputLayer.pixelStrideInBytes = sizeof(float4);
    /// Pixel format.
    outputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;
    // -------------------------------------------------------

        // --------------------------------------------------------------------
    if (denoiserOn) {
    #if OPTIX_VERSION >= 70300
      OptixDenoiserGuideLayer denoiserGuideLayer = {};

      OptixDenoiserLayer denoiserLayer = {};
      denoiserLayer.input = inputLayer;
      denoiserLayer.output = outputLayer;

      OPTIX_CHECK(optixDenoiserInvoke(myDenoiser,
                                /*stream*/0,
                                &denoiserParams,
                                (CUdeviceptr)denoiserState.get(),
                                denoiserState.size(),
                                &denoiserGuideLayer,
                                &denoiserLayer,1,
                                /*inputOffsetX*/0,
                                /*inputOffsetY*/0,
                                (CUdeviceptr)denoiserScratch.get(),
                                denoiserScratch.size()));
    #else
      OPTIX_CHECK(optixDenoiserInvoke(myDenoiser,
                                /*stream*/0,
                                &denoiserParams,
                                (CUdeviceptr)denoiserState.get(),
                                denoiserState.size(),
                                &inputLayer,1,
                                /*inputOffsetX*/0,
                                /*inputOffsetY*/0,
                                &outputLayer,
                                (CUdeviceptr)denoiserScratch.get(),
                                denoiserScratch.size()));
    #endif
    // denoisedBuffer.download((void *)this->fbPointer);
    } else {
      cudaMemcpy((void*)outputLayer.data,(void*)inputLayer.data,
                 outputLayer.width*outputLayer.height*sizeof(float4),
                 cudaMemcpyDeviceToDevice);
    }
    cudaDeviceSynchronize(); 
}

void RenderWindow::resize(const vec2i& newSize)
{
    // Resize framebuffer, and other ops (OWL::Viewer ops)
    OWLViewer::resize(newSize);

    // Resize accumulation buffer, and set to launch params
    owlBufferResize(accumBuffer, newSize.x * newSize.y);
    owlParamsSetBuffer(this->launchParams, "accumBuffer", this->accumBuffer);

    owlParamsSet2i(this->launchParams, "bufferSize", (const owl2i&)newSize);
    LOG("RESIZE!")

    if (myDenoiser) {
      OPTIX_CHECK(optixDenoiserDestroy(myDenoiser));
    }

    // ---------------------------------------------------------------------------------
    OptixDeviceContext optixContext = (OptixDeviceContext)owlContextGetOptixContext(context, 0);
    OptixDenoiserOptions denoiserOptions = {};
    #if OPTIX_VERSION >= 70300
      OPTIX_CHECK(optixDenoiserCreate(optixContext, OPTIX_DENOISER_MODEL_KIND_LDR,
                    &denoiserOptions,&myDenoiser));
    #else
      denoiserOptions.inputKind = OPTIX_DENOISER_INPUT_RGB;

    #if OPTIX_VERSION < 70100
    // these only exist in 7.0, not 7.1
      denoiserOptions.pixelFormat = OPTIX_PIXEL_FORMAT_UCHAR4;
    #endif

      OPTIX_CHECK(optixDenoiserCreate(optixContext,&denoiserOptions,&myDenoiser));
      OPTIX_CHECK(optixDenoiserSetModel(myDenoiser,OPTIX_DENOISER_MODEL_KIND_LDR,NULL,0));
    #endif

    // .. then compute and allocate memory resources for the myDenoiser
    OptixDenoiserSizes denoiserReturnSizes;
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(myDenoiser,newSize.x,newSize.y,
                                                    &denoiserReturnSizes));

    #if OPTIX_VERSION < 70100
      denoiserScratch.alloc(denoiserReturnSizes.recommendedScratchSizeInBytes);
    #else
      denoiserScratch.alloc(max(denoiserReturnSizes.withOverlapScratchSizeInBytes,
                            denoiserReturnSizes.withoutOverlapScratchSizeInBytes));
    #endif
      denoiserState.alloc(denoiserReturnSizes.stateSizeInBytes);

    this->fbSize = newSize;

    // ------------------------------------------------------------------------------------
    // resize denoisedBuffer
    denoisedBuffer.alloc(fbSize.x * fbSize.y * sizeof(float4));

    // -----------------------------------------------------------------------------------
    OPTIX_CHECK(optixDenoiserSetup(myDenoiser,0,
                                   newSize.x,newSize.y,
                                   (CUdeviceptr)denoiserState.get(),
                                   denoiserState.size(),
                                   (CUdeviceptr)denoiserScratch.get(),
                                   denoiserScratch.size()));

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