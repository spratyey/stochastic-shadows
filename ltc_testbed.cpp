#include "viewer.hpp"
#include "scene.h"
#include "common.h"
#include "flags.hpp"
#include "progressbar.hpp"

#include <chrono>
#include <fstream>
#include <filesystem>
#include <gflags/gflags.h>

namespace fs = std::filesystem;
using namespace owl;

// Compiled PTX code
extern "C" char ltc_testbed_ptx[];

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    bool isInteractive = FLAGS_isInteractive;

    std::string scenePath = FLAGS_scenePath;

    LOG("Loading scene " + scenePath);

    Scene scene;
    bool success = parseScene(scenePath, scene);
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
        fs::path savePath(FLAGS_outputPath);
        nlohmann::json stats;

        int imgName = 0;    
        for (auto cam : scene.cameras) {
            win.camera.setOrientation(cam.from, cam.at, cam.up, owl::viewer::toDegrees(acosf(cam.cosFovy)));
            win.resize(resolution);

            auto start = std::chrono::high_resolution_clock::now();

            win.accumId = 0;
#if defined(ACCUM) || defined(TEMPORAL_REUSE)
            progressbar bar(FLAGS_samples);
            for (int sample = 0; sample < FLAGS_samples; sample++) {
                bar.update();
                win.render();
            }
            std::cout << std::endl;
#else
            win.render();
#endif

            auto finish = std::chrono::high_resolution_clock::now();

            auto milliseconds_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count() / 1e6;

            std::string imgFileName = savePath.string();
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
            break;
        }
    }

    return 0;
}
