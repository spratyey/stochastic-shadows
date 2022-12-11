#pragma once

#include <fstream>
#include <vector>
#include "serialize.hpp"
#include "common.cuh"
#include "bsp.hpp"
#include "types.hpp"
#include "mesh.hpp"
#include "scene.h"
#include "model.h"
#include "bvh.hpp"
#include "silhouetteConvex.hpp"

using namespace owl;

class LightInfo {
    public:
        // Vectors to dump
        std::vector<TriLight> triLightList;
        std::vector<MeshLight> meshLightList;
        std::vector<LightBVH> lightBlas;
        std::vector<LightBVH> lightTlas;
        std::vector<LightEdge> lightEdgeList;
        std::vector<BSPNode> bspNodes;
        std::vector<int> silhouettes;
        int lightTlasHeight;

        void initialize(Scene &scene, bool calcSilhouette);
        void write(std::ofstream &stream);
        void read(std::ifstream &stream);
};