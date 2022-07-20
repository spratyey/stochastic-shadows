#include "scene.h"

void Scene::syncLights()
{
    for (auto light : this->triLights->meshes) {
        light->isLight = true;
        this->model->meshes.push_back(light);
    }
}

/*
Returns:
true - scene loaded
false - scene load failed
*/
bool parseScene(std::string sceneFile, Scene& scene)
{
    nlohmann::json sceneConfig;
    try {
        std::ifstream sceneStream(sceneFile.c_str());
        sceneStream >> sceneConfig;
    }
    catch (std::runtime_error e) {
        LOG("Could not load scene .json file");
        return false;
    }

    scene.json = sceneConfig;
    scene.jsonFilePath = sceneFile;

    // Load scene properties (spp, renderOutput, renderStatsOutput, image resolution)
    scene.spp = sceneConfig["spp"];
    scene.imgWidth = sceneConfig["width"];
    scene.imgHeight = sceneConfig["height"];
    scene.renderOutput = sceneConfig["render_output"];
    scene.renderStatsOutput = sceneConfig["render_stats"];

    // Setup different types of renderers
    try {
        for (auto renderer : sceneConfig["renderers"]) {
            scene.renderers.push_back(renderer);
        }
    }
    catch (nlohmann::json::exception e) {
        LOG("No renderers defined.");
        return false;
    }

    // Setup camera, if none present, throw an exception
    try {
        for (auto camera : sceneConfig["cameras"]) {
            SceneCamera cam;

            cam.from = vec3f(camera["from"][0], camera["from"][1], camera["from"][2]);
            cam.at = vec3f(camera["to"][0], camera["to"][1], camera["to"][2]);
            cam.up = vec3f(camera["up"][0], camera["up"][1], camera["up"][2]);
            cam.cosFovy = float(camera["cos_fovy"]);

            scene.cameras.push_back(cam);
        }
    }
    catch (nlohmann::json::exception e) {
        LOG("No cameras defined.");
        return false;
    }

    // Load .obj file of surface, if not defined, throw exception
    try {
        scene.model = loadOBJ(sceneConfig["surface_geometry"]);
    }
    catch (nlohmann::json::exception e) {
        LOG("No .obj file given/found!");
        return false;
    }

    // Load .obj file of area lights
    try {
        scene.triLights = loadOBJ(sceneConfig["area_lights"]);
        scene.syncLights();
    }
    catch (nlohmann::json::exception e) {
        LOG("No .obj file for area lights given/found!");
        return false;
    }

    return true;
}