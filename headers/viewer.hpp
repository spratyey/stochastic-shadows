#pragma once

// OWL
#include "owl/owl.h"
#include "owl/DeviceMemory.h"
#include "owl/common/math/vec.h"
#include "owl/helper/optix.h"
#include "owlViewer/OWLViewer.h"

// ImGui
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl2.h"

#include <random>
#include "scene.h"
#include "ltc_isotropic.h"
#include "silhouetteConvex.hpp"
#include "light.hpp"
#include "types.hpp"

#include "common.cuh"
#include "constants.cuh"

using namespace owl;

struct RenderWindow : public owl::viewer::OWLViewer {
    RenderWindow(Scene& scene, vec2i resolution, bool interactive, char *ptx);

    void initialize(Scene& scene, char *ptx, bool interactive);

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

    void mouseButtonLeft(const vec2i &where, bool pressed) override;
    void customKey(char key, const vec2i& pos) override;

    int getLightBVHHeight(uint32_t nodeIdx, std::vector<LightBVH>& bvh);
    float evaluateSAHForLightBVH(LightBVH& node, std::vector<TriLight>& primitives, int axis, float pos);

    template <typename T> 
    void updateLightBVHNodeBounds(uint32_t nodeIdx, std::vector<LightBVH>& bvh, std::vector<T>& primitives);

    template <typename T>
    void subdivideLightBVH(uint32_t nodeIdx, std::vector<LightBVH>& bvh, std::vector<T>& primitives);
    
    bool sbtDirty = true;

    OWLRayGen rayGen{ 0 };
    OWLRayGen spatialReuse{ 0 };
    OWLMissProg missProg{ 0 };
    
    OWLGroup world; // TLAS

    OWLContext context{ 0 };
    OWLModule module{ 0 };

    OWLParams launchParams;

    OWLBuffer accumBuffer{ 0 };
    OWLBuffer normalBuffer{ 0 };
    OWLBuffer albedoBuffer{ 0 };
    OWLBuffer binIdxBuffer { 0 };
    int accumId = 0;

    Scene currentScene;
    std::vector<SceneCamera> recordedCameras;

    std::vector<TriLight> triLightList;
    std::vector<MeshLight> meshLightList;
    
    // Create a list of edges in the mesh
    std::vector<LightEdge> lightEdgeList;

    std::vector<LightBVH> lightBlas;
    std::vector<LightBVH> lightTlas;
    int lightTlasHeight = 0;

    // Random controls
    float lerp = 0.5f;

    // Denoiser stuff
    bool denoiserOn = false;
    DeviceMemory denoisedBuffer;
    DeviceMemory denoiserScratch;
    DeviceMemory denoiserState;
    OptixDenoiser myDenoiser = nullptr;
};