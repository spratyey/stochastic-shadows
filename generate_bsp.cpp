#include <iostream>
#include <fstream>
#include <string>
#include "scene.h"
#include "light.hpp"

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./generate_bsp <light_obj_path> <output_path>" << std::endl;
        exit(1);
    }

    std::string currentScene = std::string(argv[1]);
    // std::string currentScene = "../scenes/scene_configs/bistro.json";
    LOG("Loading scene " + currentScene);
    Scene scene;
    bool success = parseScene(currentScene, scene);
    if (!success) {
        LOG("Error loading scene");
        return -1;
    } 

    // Build BVH and BSP
    LightInfo lightInfo;
    lightInfo.initialize(scene);

    // Write
    LOG("Writing light info")
    std::ofstream out_file(argv[2], std::ios::binary);
    lightInfo.write(out_file);
    out_file.close();
    LOG("Done")

    return 0;
}