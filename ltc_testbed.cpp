#include "viewer.hpp"
#include "scene.h"
#include "common.h"

#include <chrono>
#include <fstream>

using namespace owl;

// Compiled PTX code
extern "C" char ltc_testbed_ptx[];

int main(int argc, char** argv)
{
    std::string savePath;
    bool isInteractive = false;

    std::string currentScene;
    std::string defaultScene = "/home/aakashkt/ishaan/OptixRenderer/scenes/scene_configs/bistro.json";

    if (argc >= 2)
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
    RenderWindow win(scene, resolution, isInteractive, ltc_testbed_ptx);

    if (isInteractive) {
        win.camera.setOrientation(scene.cameras[0].from,
            scene.cameras[0].at,
            scene.cameras[0].up,
            owl::viewer::toDegrees(acosf(scene.cameras[0].cosFovy)));
        win.enableFlyMode();
        win.setWorldScale(length(scene.model->bounds.span()));
        win.enableInspectMode(owl::box3f(scene.model->bounds.lower, scene.model->bounds.upper));

        // ##################################################################
        // now that everything is ready: launch it ....
        // ##################################################################
        win.showAndRun();
    } else {
        if (argc == 4) {
            savePath = std::string(argv[3]);
        } else {
            savePath = "output";
        }

        nlohmann::json stats;

        int imgName = 0;    
        for (auto cam : scene.cameras) {
            win.camera.setOrientation(cam.from, cam.at, cam.up, owl::viewer::toDegrees(acosf(cam.cosFovy)));
            win.resize(resolution);

            auto start = std::chrono::high_resolution_clock::now();

            win.accumId = 0;
#if RENDERER == DIRECT_LIGHTING
            for (int sample = 0; sample < scene.spp; sample++) {
                win.render();
            }
#else
            win.render();
#endif

            auto finish = std::chrono::high_resolution_clock::now();

            auto milliseconds_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count() / 1e6;

            std::string imgFileName = savePath + "/" + "LTC_TestBed" + "_" + std::to_string(imgName) + ".png";
            nlohmann::json currentStats = {
                {"image_name", imgFileName},
                {"width", scene.imgWidth},
                {"height", scene.imgHeight},
                {"frametime_milliseconds", milliseconds_taken},
                {"num_area_lights", win.triLightList.size()},
            };

            stats.push_back(currentStats);

            win.screenShot(imgFileName);
            imgName++;
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
