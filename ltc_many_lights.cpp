#include "viewer.hpp"

#include <chrono>
#include <fstream>

using namespace owl;

// Compiled PTX code
extern "C" char ltc_many_lights_cuda_ptx[];

const char* rendererNames[NUM_RENDERER_TYPES] = {"Diffuse", "Alpha", "Normals", "Silhouette",
												"Direct Light (Light)", "Direct Light (BRDF)", "Direct Light (MIS)",
												"Direct Light (Light BVH) (Light)", "Direct Light (Light BVH) (BRDF)", "Direct Light (Light BVH) (MIS)",
												"LTC Baseline", "LTC (Light BVH, Linear)", "LTC (Light BVH, BST)", "LTC (Light BVH, Silhoutte)"};

int main(int argc, char** argv)
{
    std::string savePath;
    bool isInteractive = false;

    std::string currentScene;
    std::string defaultScene = "/home/aakashkt/ishaan/OptixRenderer/scenes/scene_configs/silhoutte_test.json";

    if (argc == 2)
        currentScene = std::string(argv[1]);
    else
        currentScene = defaultScene;

    if (argc >= 3) {
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
    RenderWindow win(scene, resolution, isInteractive, ltc_many_lights_cuda_ptx);

    if (isInteractive) {
        win.camera.setOrientation(scene.cameras[0].from,
            scene.cameras[0].at,
            scene.cameras[0].up,
            owl::viewer::toDegrees(acosf(scene.cameras[0].cosFovy)));
        win.enableFlyMode();
        win.enableInspectMode(owl::box3f(scene.model->bounds.lower, scene.model->bounds.upper));
        win.setWorldScale(length(scene.model->bounds.span()));
        win.setRendererType(static_cast<RendererType>(4));

        // ##################################################################
        // now that everything is ready: launch it ....
        // ##################################################################
        win.showAndRun();
    }
    else {
        if (argc == 4) {
          savePath = std::string(argv[3]);
        } else {
          savePath = "output";
        }


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
                // Samples should be 1 for LTC
                int samples = (renderer > 8) ? 1 : scene.spp;
                for (int sample = 0; sample < samples; sample++) {
                    win.render();
                }

                auto finish = std::chrono::high_resolution_clock::now();

                auto milliseconds_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count() / 1e6;

                std::string imgFileName = savePath + "/" + rendererName + "_" + std::to_string(imgName) + ".png";
                nlohmann::json currentStats = {
                    {"image_name", imgFileName},
                    {"spp", samples},
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
        for (auto stat : stats) {
          LOG(stat["image_name"]);
          LOG(stat["frametime_milliseconds"]);
        }
    }

    return 0;
}
