#include <iostream>
#include "set.cuh"
#include "lcg_random.cuh"
#include "scene.h"
#include "types.hpp"
#include "bvh.hpp"
#include "constants.cuh"

using namespace owl;

__constant__ int bvhHeight;

__device__
void stochasticTraverseLBVHNoDup(LightBVH *bvh, Set *set, vec2i rand, int &selectedIdx, vec3f p) {
    selectedIdx = -1;

    float r1 = rand.x;
    float r2 = rand.y;

    int nodeIdx = 0;
    bool prevNodeStatus = false;
    int toInsert = -1;

    for (int i = 0; i <= bvhHeight; i++) {
        LightBVH node = bvh[nodeIdx];

        // If leaf
        if (node.left == 0 && node.right == 0) {
            if (node.primCount == 1) {
                selectedIdx = node.primIdx;
            }
            // TODO: Figure out how to has primIdx
            else {
                selectedIdx = node.primIdx + round(r1 * (node.primCount-1));
            }
      
            if (!prevNodeStatus) {
                toInsert = nodeIdx;
            }
            break;
        }

        LightBVH leftNode = bvh[node.left];
        LightBVH rightNode = bvh[node.right];

        float leftImp = leftNode.flux / pow(length(leftNode.aabbMid - p), 2.f);
        float rightImp = rightNode.flux / pow(length(rightNode.aabbMid - p), 2.f);
        float sum = leftImp + rightImp;

        leftImp = leftImp / sum;
        rightImp = rightImp / sum;

        float eps = r2;
        bool selected = false; // Select left node by default
        if (eps > leftImp) {
            selected = true;  // Select right node
        }
  
        // Choose other node if one is taken
        bool leftFull = set->exists(node.left);
        bool rightFull = set->exists(node.right);
        if ((!selected && leftFull) || (selected && rightFull)) {
            selected = !selected;
        }
  
        // Book-keeping
        bool curStatus = leftFull || rightFull;
        if (curStatus && !prevNodeStatus) {
            toInsert = nodeIdx;
        }
        prevNodeStatus = curStatus;
  
        // Set selected nodeIdx
        nodeIdx = selected ? node.right : node.left;

        // Generate new random numbers
        if (r1 < leftImp) {
            r1 = r1 / leftImp;
        } else {
            r1 = (r1 - leftImp) / rightImp;
        }

        // Generate new random numbers
        if (r2 < leftImp) {
            r2 = r2 / leftImp;
        } else {
            r2 = (r2 - leftImp) / rightImp;
        }
    }

    set->insert(toInsert);
}

__global__
void test_bvh(LightBVH *bvh) {
    Set selectedSet;
    int selectedIdx;
    for (int i = 0; i < MAX_LTC_LIGHTS; i++) {
        stochasticTraverseLBVHNoDup(bvh, &selectedSet, vec2i(0, 0), selectedIdx, vec3f(0));
    }
}

int main() {
    // Load the scene
    std::string scenePath = "/home/aakashkt/ishaan/OptixRenderer/scenes/scene_configs/test_scene.json";
    Scene scene;
    bool success = parseScene(scenePath, scene);

    // Create list of all lights with their area and intensity
    std::vector<MeshLight> meshLightList;
    for (auto light : scene.triLights->meshes) {
        MeshLight meshLight;
        meshLight.flux = 0.f;

        int numTri = 0;
        float totalArea = 0;
        for (auto index : light->index) {
            TriLight triLight;

            triLight.v1 = light->vertex[index.x];
            triLight.v2 = light->vertex[index.y];
            triLight.v3 = light->vertex[index.z];

            triLight.cg = (triLight.v1 + triLight.v2 + triLight.v3) / 3.f;
            triLight.normal = normalize(light->normal[index.x] + light->normal[index.y] + light->normal[index.z]);
            triLight.area = 0.5f * length(cross(triLight.v1 - triLight.v2, triLight.v3 - triLight.v2));

            triLight.emit = light->emit;
            triLight.flux = triLight.area * length(triLight.emit);

            triLight.aabbMin = owl::min(triLight.aabbMin, triLight.v1);
            triLight.aabbMin = owl::min(triLight.aabbMin, triLight.v2);
            triLight.aabbMin = owl::min(triLight.aabbMin, triLight.v3);

            triLight.aabbMax = owl::max(triLight.aabbMax, triLight.v1);
            triLight.aabbMax = owl::max(triLight.aabbMax, triLight.v2);
            triLight.aabbMax = owl::max(triLight.aabbMax, triLight.v3);

            // Next, update the AABB and flux of current light mesh
            meshLight.aabbMin = owl::min(meshLight.aabbMin, triLight.aabbMin);
            meshLight.aabbMax = owl::max(meshLight.aabbMax, triLight.aabbMax);
            meshLight.flux += triLight.flux;
            
            // Set average emmitance weighted by triangle size
            meshLight.avgEmit += triLight.area * light->emit;

            // Keep track of number of triangles in the current light mesh
            numTri++;

            // Keep track of total triangle area
            totalArea += triLight.area;
        }

        meshLight.avgEmit /= totalArea;

        // Insert spans 
        meshLight.triCount = numTri;

        meshLight.cg = (meshLight.aabbMin + meshLight.aabbMax) / 2.f;

        meshLightList.push_back(meshLight);
    }

    // Build BVH
    std::vector<LightBVH> lightTlas;

    LightBVH root;
    root.left = root.right = 0;
    root.primCount = meshLightList.size();
    lightTlas.push_back(root);

    updateLightBVHNodeBounds<MeshLight>(0, lightTlas, meshLightList);
    subdivideLightBVH<MeshLight>(0, lightTlas, meshLightList);
    int lightTlasHeight = getLightBVHHeight(0, lightTlas);

    std::cout << "Build BVH with " << lightTlas.size() << " nodes" << std::endl;
    std::cout << "Height of BVH " << lightTlasHeight << std::endl;

    // Create CUDA arrays to store BVH
    LightBVH* dLightTlas;
    cudaMalloc(&dLightTlas, lightTlas.size() * sizeof(LightBVH));

    // Copy BVH height to constant memory
    cudaMemcpyToSymbol(bvhHeight, &lightTlasHeight, sizeof(int));

    cudaMemcpy(dLightTlas, lightTlas.data(), lightTlas.size() * sizeof(LightBVH), cudaMemcpyHostToDevice);

    // Calculate block size
    int resolution = 1920*1080;
    int threadCount = 256;
    int blockSize = ceil((float)resolution / (float)threadCount);

    std::cout << "Choosing " << MAX_LTC_LIGHTS << std::endl;
    std::cout << "Launching with " << blockSize << " blocks with " << threadCount << " threads per block" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    // Launch kernel
    test_bvh<<<blockSize,threadCount>>>(dLightTlas);
    
    // Wait for it to complete
    cudaDeviceSynchronize();

    auto finish = std::chrono::high_resolution_clock::now();
    auto milliseconds_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count() / 1e6;

    std::cout << "Time taken: " << milliseconds_taken << "ms" << std::endl;
}