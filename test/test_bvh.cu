#include <iostream>
#include "set.cuh"
#include "lcg_random.cuh"
#include "scene.h"
#include "types.hpp"
#include "bvh.hpp"
#include "constants.cuh"

#ifdef DEBUG_BUILD
#define print(bIdx, tIdx, ...) { if (blockIdx.x == bIdx && threadIdx.x == tIdx) { printf( __VA_ARGS__ ); } }
#else
#define print(bIdx, tIdx, ...)
#endif

using namespace owl;

__constant__ int bvhHeight;
__constant__ vec3f boundMin;
__constant__ vec3f boundMax;

__device__
void stochasticTraverseLBVH(LightBVH* bvh, vec2f rand, int& selectedIdx, vec3f p) {
    selectedIdx = -1;

    float r1 = rand.x;
    float r2 = rand.y;

    int nodeIdx = 0;
    for (int i = 0; i < bvhHeight + 1; i++) {
        LightBVH node = bvh[nodeIdx];

        // If leaf
        if (node.left == 0 && node.right == 0) {
            if (node.primCount == 1) {
                selectedIdx = node.primIdx;
            }
            else {
                selectedIdx = node.primIdx + round(r1 * (node.primCount-1));
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
        if (eps < leftImp) {
            nodeIdx = node.left;
        } else {
            nodeIdx = node.right;
        }

        // Generate new random numbers
        if (r1 < leftImp)
            r1 = r1 / leftImp;
        else
            r1 = (r1 - leftImp) / rightImp;

        // Generate new random numbers
        if (r2 < leftImp)
            r2 = r2 / leftImp;
        else
            r2 = (r2 - leftImp) / rightImp;
    }
}

__device__
void stochasticTraverseLBVHNoDup(LightBVH *bvh, Set *set, vec2f rand, int &selectedIdx, vec3f p, int &checkHeight) {
    selectedIdx = -1;

    float r1 = rand.x;
    float r2 = rand.y;

    int nodeIdx = 0;
    bool prevNodeStatus = false;
    int toInsert = -1;
    int toInsertHeight = -1;

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
                toInsertHeight = i;
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
        if (i >= checkHeight) {
            bool leftFull = set->exists(node.left);
            bool rightFull = set->exists(node.right);
            if ((!selected && leftFull) || (selected && rightFull)) {
                selected = !selected;
            }
    
            // Book-keeping
            bool curStatus = leftFull || rightFull;
            if (curStatus && !prevNodeStatus) {
                toInsert = nodeIdx;
                toInsertHeight = i;
            }
            prevNodeStatus = curStatus;
        }
  
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

    checkHeight = min(checkHeight, toInsertHeight - 1);
    set->insert(toInsert);
}

__global__
void test_bvh_no_dup_rejection(LightBVH *bvh, int *chosen) {
    Set selectedSet;
    int selectedIdx;
    int selectedLights[MAX_LTC_LIGHTS];
    int selectedCount = 0;
    LCGRand rng = get_rng(10007, make_uint2(blockIdx.x, threadIdx.x), make_uint2(gridDim.x, blockDim.x));
    vec3f rand = vec3f(lcg_randomf(rng), lcg_randomf(rng), lcg_randomf(rng));
    vec3f p = boundMin + rand * (boundMax - boundMin);
    for (int i = 0; i < 2*MAX_LTC_LIGHTS; i++) {
        if (selectedCount == MAX_LTC_LIGHTS) {
            break;
        }
        stochasticTraverseLBVH(bvh, vec2f(lcg_randomf(rng), lcg_randomf(rng)), selectedIdx, vec3f(0));
        if (!selectedSet.exists(selectedIdx)) {
            selectedSet.insert(selectedIdx);
            selectedLights[selectedCount++] = selectedIdx;
        }
    }
    int offset = (threadIdx.x + blockDim.x*blockIdx.x)*MAX_LTC_LIGHTS;
    for (int i = 0; i < selectedCount; i++) {
        chosen[offset+i] = selectedLights[i];
    }
}

__global__
void test_bvh_no_dup(LightBVH *bvh, int *chosen) {
    Set selectedSet;
    int selectedIdx;
    int selectedLights[MAX_LTC_LIGHTS];
    int selectedCount = 0;
    LCGRand rng = get_rng(10007, make_uint2(blockIdx.x, threadIdx.x), make_uint2(gridDim.x, blockDim.x));
    vec3f rand = vec3f(lcg_randomf(rng), lcg_randomf(rng), lcg_randomf(rng));
    vec3f p = boundMin + rand * (boundMax - boundMin);
    int checkHeight = bvhHeight;
    for (int i = 0; i < MAX_LTC_LIGHTS; i++) {
        stochasticTraverseLBVHNoDup(bvh, &selectedSet, vec2f(lcg_randomf(rng), lcg_randomf(rng)), selectedIdx, vec3f(0), checkHeight);
        selectedLights[selectedCount++] = selectedIdx;
    }
    int offset = (threadIdx.x + blockDim.x*blockIdx.x)*MAX_LTC_LIGHTS;
    for (int i = 0; i < selectedCount; i++) {
        chosen[offset+i] = selectedLights[i];
    }
}

__global__
void test_bvh_dup(LightBVH *bvh, int *chosen) {
    int selectedIdx;
    int selectedLights[MAX_LTC_LIGHTS];
    int selectedCount = 0;
    LCGRand rng = get_rng(10007, make_uint2(blockIdx.x, threadIdx.x), make_uint2(gridDim.x, blockDim.x));
    vec3f rand = vec3f(lcg_randomf(rng), lcg_randomf(rng), lcg_randomf(rng));
    vec3f p = boundMin + rand * (boundMax - boundMin);
    for (int i = 0; i < MAX_LTC_LIGHTS; i++) {
        stochasticTraverseLBVH(bvh, vec2f(lcg_randomf(rng), lcg_randomf(rng)), selectedIdx, vec3f(0));
        selectedLights[selectedCount++] = selectedIdx;
    }
    int offset = (threadIdx.x + blockDim.x*blockIdx.x)*MAX_LTC_LIGHTS;
    for (int i = 0; i < selectedCount; i++) {
        chosen[offset+i] = selectedLights[i];
    }
}

int main() {
    // Load the scene
    std::string scenePath = "/home/aakashkt/ishaan/OptixRenderer/scenes/scene_configs/bistro.json";
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
    root.primIdx = 0;
    root.primCount = meshLightList.size();
    lightTlas.push_back(root);

    updateLightBVHNodeBounds<MeshLight>(0, lightTlas, meshLightList);
    subdivideLightBVH<MeshLight>(0, lightTlas, meshLightList);

    std::vector<LightBVH> reorderLightTlas;
    reorderLightBVH(lightTlas, reorderLightTlas);

    for (int i = 0; i < lightTlas.size(); i++) {
        lightTlas[i] = reorderLightTlas[i];
    }

    int lightTlasHeight = getLightBVHHeight(0, lightTlas);

    std::cout << "Build BVH with " << lightTlas.size() << " nodes" << std::endl;
    std::cout << "Height of BVH " << lightTlasHeight << std::endl;

    std::cout << "Scene Bounds (min): " << lightTlas[0].aabbMin.x << " " << lightTlas[0].aabbMin.y << " " << lightTlas[0].aabbMin.z << std::endl;
    std::cout << "Scene Bounds (max): " << lightTlas[0].aabbMax.x << " " << lightTlas[0].aabbMax.y << " " << lightTlas[0].aabbMax.z << std::endl;

    // Calculate block size
    int resolution = 1920*1080;
    int threadCount = 256;
    int blockSize = ceil((float)resolution / (float)threadCount);

    // Create CUDA arrays to store BVH
    LightBVH* dLightTlas;
    cudaMalloc(&dLightTlas, lightTlas.size() * sizeof(LightBVH));

    // Create array to store chosen lights
    std::vector<int> chosenLights(blockSize * threadCount * MAX_LTC_LIGHTS, -1);
    int *dChosenLights;
    cudaMalloc(&dChosenLights, chosenLights.size() * sizeof(int));

    // Copy BVH height to constant memory
    cudaMemcpyToSymbol(bvhHeight, &lightTlasHeight, sizeof(int));
    cudaMemcpyToSymbol(boundMin, &lightTlas[0].aabbMin, sizeof(vec3f));
    cudaMemcpyToSymbol(boundMax, &lightTlas[0].aabbMax, sizeof(vec3f));

    cudaMemcpy(dLightTlas, lightTlas.data(), lightTlas.size() * sizeof(LightBVH), cudaMemcpyHostToDevice);

    std::cout << "Choosing " << MAX_LTC_LIGHTS << " lights" << std::endl;
    std::cout << "Launching with " << blockSize << " blocks with " << threadCount << " threads per block" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    // Launch kernel
    test_bvh_no_dup<<<blockSize,threadCount>>>(dLightTlas, dChosenLights);
    // Wait for kernel to finish executing
    cudaDeviceSynchronize();
    auto finish = std::chrono::high_resolution_clock::now();
    auto milliseconds_taken = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
    std::cout << "Time taken for no duplicates: " << milliseconds_taken << "ms" << std::endl;
    // Copy back the chosen lights
    cudaMemcpy(chosenLights.data(), dChosenLights, chosenLights.size() * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Lights chosen: ";
    for (int i = 0; i < MAX_LTC_LIGHTS; i++) {
        std::cout << chosenLights[i] << ", ";
    }
    std::cout << std::endl;

    start = std::chrono::high_resolution_clock::now();
    // Launch kernel
    test_bvh_no_dup_rejection<<<blockSize,threadCount>>>(dLightTlas, dChosenLights);
    // Wait for kernel to finish executing
    cudaDeviceSynchronize();
    finish = std::chrono::high_resolution_clock::now();
    milliseconds_taken = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
    std::cout << "Time taken for no duplicates (rejection sampling): " << milliseconds_taken << "ms" << std::endl;
    // Copy back the chosen lights
    cudaMemcpy(chosenLights.data(), dChosenLights, chosenLights.size() * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Lights chosen: ";
    for (int i = 0; i < MAX_LTC_LIGHTS; i++) {
        std::cout << chosenLights[i] << ", ";
    }
    std::cout << std::endl;

    start = std::chrono::high_resolution_clock::now();
    // Launch kernel
    test_bvh_dup<<<blockSize,threadCount>>>(dLightTlas, dChosenLights);
    // Wait for kernel to finish executing
    cudaDeviceSynchronize();
    finish = std::chrono::high_resolution_clock::now();
    milliseconds_taken = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
    std::cout << "Time taken for duplicates: " << milliseconds_taken << "ms" << std::endl;
    // Copy back the chosen lights
    cudaMemcpy(chosenLights.data(), dChosenLights, chosenLights.size() * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Lights chosen: ";
    for (int i = 0; i < MAX_LTC_LIGHTS; i++) {
        std::cout << chosenLights[i] << ", ";
    }
    std::cout << std::endl;

    cudaFree(dLightTlas);
    cudaFree(dChosenLights);
}