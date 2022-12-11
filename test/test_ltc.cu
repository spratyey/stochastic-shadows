#include <iostream>
#include "owl/common/math/vec.h"
#include "owl/owl.h"
#include "lcg_random.cuh"
#include "scene.h"
#include "types.hpp"
#include "constants.cuh"
#include "polygon_utils.cuh"
#include "ltc_isotropic.h"

#ifdef DEBUG_BUILD
#define print(bIdx, tIdx, ...) { if (blockIdx.x == bIdx && threadIdx.x == tIdx) { printf( __VA_ARGS__ ); } }
#else
#define print(bIdx, tIdx, ...)
#endif

using namespace owl;

__constant__ int meshLightSize;
__constant__ int triLightSize;
__constant__ cudaTextureObject_t ltc_1;
__constant__ cudaTextureObject_t ltc_2;
__constant__ cudaTextureObject_t ltc_3;

__device__
vec3f apply_mat(vec3f mat[3], vec3f v) {
    vec3f result(dot(mat[0], v), dot(mat[1], v), dot(mat[2], v));
    return result;
}

__device__
void matrixTranspose(vec3f m[3], vec3f mTrans[3]) {
    mTrans[0] = m[0];
    mTrans[1] = m[1];
    mTrans[2] = m[2];

    mTrans[1].x = m[0].y;
    mTrans[2].x = m[0].z;

    mTrans[0].y = m[1].x;
    mTrans[2].y = m[1].z;

    mTrans[0].z = m[2].x;
    mTrans[1].z = m[2].y;
}

__device__
void orthonormalBasis(vec3f n, vec3f mat[3], vec3f invmat[3]) {
    vec3f c1, c2, c3;
    if (n.z < -0.999999f)
    {
        c1 = vec3f(0, -1, 0);
        c2 = vec3f(-1, 0, 0);
    }
    else
    {
        float a = 1. / (1. + n.z);
        float b = -n.x * n.y * a;
        c1 = normalize(vec3f(1. - n.x * n.x * a, b, -n.x));
        c2 = normalize(vec3f(b, 1. - n.y * n.y * a, -n.y));
    }
    c3 = n;

    mat[0] = c1;
    mat[1] = c2;
    mat[2] = c3;

    matrixTranspose(mat, invmat);
}

__device__
vec3f integrateEdgeVec(vec3f v1, vec3f v2)
{
    float x = dot(v1, v2);
    float y = abs(x);

    float a = 0.8543985f + (0.4965155f + 0.0145206f * y) * y;
    float b = 3.4175940f + (4.1616724f + y) * y;
    float v = a / b;

    float theta_sintheta = (x > 0.0f) ? v : 0.5 * (1.0f / sqrt(max(1.0f - x * x, 1e-7))) - v;

    return cross(v1, v2) * theta_sintheta;
}

__device__
float integrateEdge(vec3f v1, vec3f v2) {
    return integrateEdgeVec(v1, v2).z;
}

// Returns a vector c such that atan2(a.y, a.x) + atan2(b.y, b.x) = atan2(c.y, c.x)
__device__
vec2f atan2Sum(vec2f a, vec2f b) {
	// Readable version:
	return vec2f(a.x * b.x - a.y * b.y, a.y * b.x + a.x * b.y);
}
// Returns a vector c such that atan2(a.y, a.x) - atan2(b.y, b.x) = atan2(c.y, c.x)
__device__
vec2f atan2Diff(vec2f a, vec2f b) {
	// Readable version:
	return vec2f(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y);
}

// Slightly faster version that's actually used:
__device__
float integrateEdgeSil(vec3f a, vec3f b) {
	vec3f c = cross(a, b);
	vec2f n = vec2f(rsqrt(dot(a, a) * dot(b, b)), rsqrt(dot(c, c)));
	float AdotB = owl::clamp(dot(a, b) * n.x, -1.0f, 1.0f);
	return acos(AdotB) * c.z * n.y;
}

__device__
vec3f equatorInteresection(vec3f a, vec3f b) {
    float t = a.z / (a.z - b.z);
    return normalize(a + (b - a) * t);
}

__device__
vec3f integrateOverPolygon(vec3f p, vec3f ltc_mat_inv[3], vec3f iso_frame[3], vec3f to_local[3], float amplitude, TriLight& triLight, vec3f diffuse) {
    vec3f lv1 = triLight.v1;
    vec3f lv2 = triLight.v2;
    vec3f lv3 = triLight.v3;
    vec3f lemit = triLight.emit;

    // Move to origin and normalize
    lv1 = owl::normalize(lv1 - p);
    lv2 = owl::normalize(lv2 - p);
    lv3 = owl::normalize(lv3 - p);
    // print(0, 0, "%f %f %f\n", lv1.x, lv1.y, lv1.z);

    vec3f cg = normalize(lv1 + lv2 + lv3);

    lv1 = owl::normalize(apply_mat(to_local, lv1));
    lv2 = owl::normalize(apply_mat(to_local, lv2));
    lv3 = owl::normalize(apply_mat(to_local, lv3));

    lv1 = owl::normalize(apply_mat(iso_frame, lv1));
    lv2 = owl::normalize(apply_mat(iso_frame, lv2));
    lv3 = owl::normalize(apply_mat(iso_frame, lv3));

    float diffuse_shading = 0.f;
    float ggx_shading = 0.f;

    vec3f diff_clipped[5] = { lv1, lv2, lv3, lv1, lv1 };
    // int diff_vcount = clipPolygon(3, diff_clipped);
    int diff_vcount = 3; 
    
    if (diff_vcount == 3) {
        diffuse_shading = integrateEdge(diff_clipped[0], diff_clipped[1]);
        diffuse_shading += integrateEdge(diff_clipped[1], diff_clipped[2]);
        diffuse_shading += integrateEdge(diff_clipped[2], diff_clipped[0]);
        diffuse_shading = owl::abs(diffuse_shading);
    }
    else if (diff_vcount == 4) {
        diffuse_shading = integrateEdge(diff_clipped[0], diff_clipped[1]);
        diffuse_shading += integrateEdge(diff_clipped[1], diff_clipped[2]);
        diffuse_shading += integrateEdge(diff_clipped[2], diff_clipped[3]);
        diffuse_shading += integrateEdge(diff_clipped[3], diff_clipped[0]);
        diffuse_shading = owl::abs(diffuse_shading);
    }

    diff_clipped[0] = owl::normalize(apply_mat(ltc_mat_inv, lv1));
    diff_clipped[1] = owl::normalize(apply_mat(ltc_mat_inv, lv2));
    diff_clipped[2] = owl::normalize(apply_mat(ltc_mat_inv, lv3));
    diff_clipped[3] = owl::normalize(apply_mat(ltc_mat_inv, lv1));
    diff_clipped[4] = owl::normalize(apply_mat(ltc_mat_inv, lv1));

    vec3f ltc_clipped[5] = { diff_clipped[0], diff_clipped[1], diff_clipped[2], diff_clipped[3], diff_clipped[4] };
    // int ltc_vcount = clipPolygon(diff_vcount, ltc_clipped);
    int ltc_vcount = 3; 

    if (ltc_vcount == 3) {
        ggx_shading = integrateEdge(ltc_clipped[0], ltc_clipped[1]);
        ggx_shading += integrateEdge(ltc_clipped[1], ltc_clipped[2]);
        ggx_shading += integrateEdge(ltc_clipped[2], ltc_clipped[0]);
        ggx_shading = owl::abs(ggx_shading);
    }
    else if (ltc_vcount == 4) {
        ggx_shading = integrateEdge(ltc_clipped[0], ltc_clipped[1]);
        ggx_shading += integrateEdge(ltc_clipped[1], ltc_clipped[2]);
        ggx_shading += integrateEdge(ltc_clipped[2], ltc_clipped[3]);
        ggx_shading += integrateEdge(ltc_clipped[3], ltc_clipped[0]);
        ggx_shading = owl::abs(ggx_shading);
    }
    else if (ltc_vcount == 5) {
        ggx_shading = integrateEdge(ltc_clipped[0], ltc_clipped[1]);
        ggx_shading += integrateEdge(ltc_clipped[1], ltc_clipped[2]);
        ggx_shading += integrateEdge(ltc_clipped[2], ltc_clipped[3]);
        ggx_shading += integrateEdge(ltc_clipped[3], ltc_clipped[4]);
        ggx_shading += integrateEdge(ltc_clipped[4], ltc_clipped[0]);
        ggx_shading = owl::abs(ggx_shading);
    }

    vec3f color = (diffuse * lemit * diffuse_shading) + (amplitude * lemit * ggx_shading);
    return color;
}

__global__
void test_ltc_rand(MeshLight *meshLights, TriLight *triLights, vec3f *result) {
    LCGRand rng = get_rng(10007, make_uint2(blockIdx.x, threadIdx.x), make_uint2(gridDim.x, blockDim.x));

    // vec3f p = vec3f(2500*lcg_randomf(rng), 2500*lcg_randomf(rng), 500*lcg_randomf(rng));
    vec3f p = vec3f(lcg_randomf(rng), lcg_randomf(rng), lcg_randomf(rng));
    vec3f normal = normalize(vec3f(lcg_randomf(rng), lcg_randomf(rng), lcg_randomf(rng)));
    vec3f diffuse = vec3f(100*lcg_randomf(rng), 100*lcg_randomf(rng), 100*lcg_randomf(rng));
    vec3f wo_local = normalize(vec3f(lcg_randomf(rng), lcg_randomf(rng), lcg_randomf(rng)));
    vec3f to_world[3], to_local[3], iso_frame[3], ltc_mat_inv[3];
    orthonormalBasis(normal, to_local, to_world);

    iso_frame[0] = wo_local;
    iso_frame[0].z = 0.f;
    iso_frame[0] = normalize(iso_frame[0]);
    iso_frame[2] = vec3f(0, 0, 1);
    iso_frame[1] = normalize(cross(iso_frame[2], iso_frame[0]));

    ltc_mat_inv[0] = normalize(vec3f(lcg_randomf(rng), lcg_randomf(rng), lcg_randomf(rng)));
    ltc_mat_inv[1] = normalize(vec3f(lcg_randomf(rng), lcg_randomf(rng), lcg_randomf(rng)));
    ltc_mat_inv[2] = normalize(vec3f(lcg_randomf(rng), lcg_randomf(rng), lcg_randomf(rng)));
    float amplitude = lcg_randomf(rng);

    int offset = threadIdx.x + blockDim.x*blockIdx.x;
    result[offset] = vec3f(0);
    for (int i = 0; i < MAX_ELEMS; i += 1) {
        int chosenIdx = lcg_randomf(rng) * triLightSize;
        TriLight triLight = triLights[chosenIdx];
        result[offset] += integrateOverPolygon(p, ltc_mat_inv, iso_frame, to_local, amplitude, triLight, diffuse);
    }
}

__global__
void test_ltc_seq(MeshLight *meshLights, TriLight *triLights, vec3f *result) {
    LCGRand rng = get_rng(10007, make_uint2(blockIdx.x, threadIdx.x), make_uint2(gridDim.x, blockDim.x));

    vec3f p = vec3f(lcg_randomf(rng), lcg_randomf(rng), lcg_randomf(rng));
    vec3f normal = normalize(vec3f(lcg_randomf(rng), lcg_randomf(rng), lcg_randomf(rng)));
    vec3f diffuse = vec3f(100*lcg_randomf(rng), 100*lcg_randomf(rng), 100*lcg_randomf(rng));
    vec3f wo_local = normalize(vec3f(lcg_randomf(rng), lcg_randomf(rng), lcg_randomf(rng)));
    vec3f to_world[3], to_local[3], iso_frame[3], ltc_mat_inv[3];
    orthonormalBasis(normal, to_local, to_world);

    iso_frame[0] = wo_local;
    iso_frame[0].z = 0.f;
    iso_frame[0] = normalize(iso_frame[0]);
    iso_frame[2] = vec3f(0, 0, 1);
    iso_frame[1] = normalize(cross(iso_frame[2], iso_frame[0]));

    ltc_mat_inv[0] = normalize(vec3f(lcg_randomf(rng), lcg_randomf(rng), lcg_randomf(rng)));
    ltc_mat_inv[1] = normalize(vec3f(lcg_randomf(rng), lcg_randomf(rng), lcg_randomf(rng)));
    ltc_mat_inv[2] = normalize(vec3f(lcg_randomf(rng), lcg_randomf(rng), lcg_randomf(rng)));
    float amplitude = lcg_randomf(rng);

    int offset = threadIdx.x + blockDim.x*blockIdx.x;
    result[offset] = vec3f(0);
    for (int i = 0; i < MAX_ELEMS; i += 1) {
        TriLight triLight = triLights[(offset +  i) % triLightSize];
        result[offset] += integrateOverPolygon(p, ltc_mat_inv, iso_frame, to_local, amplitude, triLight, diffuse);
    }
}

__global__
void test_ltc_all(MeshLight *meshLights, TriLight *triLights, vec3f *result) {
    LCGRand rng = get_rng(10007, make_uint2(blockIdx.x, threadIdx.x), make_uint2(gridDim.x, blockDim.x));

    // vec3f p = vec3f(2500*lcg_randomf(rng), 2500*lcg_randomf(rng), 500*lcg_randomf(rng));
    vec3f p = vec3f(lcg_randomf(rng), lcg_randomf(rng), lcg_randomf(rng));
    vec3f normal = normalize(vec3f(lcg_randomf(rng), lcg_randomf(rng), lcg_randomf(rng)));
    vec3f diffuse = vec3f(100*lcg_randomf(rng), 100*lcg_randomf(rng), 100*lcg_randomf(rng));
    vec3f wo_local = normalize(vec3f(lcg_randomf(rng), lcg_randomf(rng), lcg_randomf(rng)));
    vec3f to_world[3], to_local[3], iso_frame[3], ltc_mat_inv[3];
    orthonormalBasis(normal, to_local, to_world);

    iso_frame[0] = wo_local;
    iso_frame[0].z = 0.f;
    iso_frame[0] = normalize(iso_frame[0]);
    iso_frame[2] = vec3f(0, 0, 1);
    iso_frame[1] = normalize(cross(iso_frame[2], iso_frame[0]));

    ltc_mat_inv[0] = normalize(vec3f(lcg_randomf(rng), lcg_randomf(rng), lcg_randomf(rng)));
    ltc_mat_inv[1] = normalize(vec3f(lcg_randomf(rng), lcg_randomf(rng), lcg_randomf(rng)));
    ltc_mat_inv[2] = normalize(vec3f(lcg_randomf(rng), lcg_randomf(rng), lcg_randomf(rng)));
    float amplitude = lcg_randomf(rng);

    int offset = threadIdx.x + blockDim.x*blockIdx.x;
    result[offset] = vec3f(0);
    for (int i = 0; i < triLightSize; i += 1) {
        TriLight triLight = triLights[i];
        result[offset] += integrateOverPolygon(p, ltc_mat_inv, iso_frame, to_local, amplitude, triLight, diffuse);
    }
}

int main() {
    // Load the scene
    std::string scenePath = "/home/aakashkt/ishaan/OptixRenderer/scenes/scene_configs/bistro.json";
    Scene scene;
    bool success = parseScene(scenePath, scene);

    // Create list of all lights with their area and intensity
    std::vector<MeshLight> meshLightList;
    std::vector<TriLight> triLightList;
    for (auto light : scene.triLights->meshes) {
        MeshLight meshLight;
        meshLight.flux = 0.f;
        meshLight.triIdx = triLightList.size();
        meshLight.triStartIdx = triLightList.size();

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

            triLightList.push_back(triLight);
        }

        meshLight.avgEmit /= totalArea;

        // Insert spans 
        meshLight.triCount = numTri;

        meshLight.cg = (meshLight.aabbMin + meshLight.aabbMax) / 2.f;

        meshLightList.push_back(meshLight);
    }


    // Calculate block size
    int resolution = 1920*1080;
    int threadCount = 256;
    int blockSize = ceil((float)resolution / (float)threadCount);

    // Create CUDA arrays to store lights
    MeshLight* dMeshLights;
    cudaMalloc(&dMeshLights, meshLightList.size() * sizeof(MeshLight));

    // Create CUDA arrays to store actual triangles
    TriLight* dTriLights;
    cudaMalloc(&dTriLights, triLightList.size() * sizeof(TriLight));

    // Create array to store integrated answer
    std::vector<vec3f> totalIntegral(blockSize * threadCount, 0.0);
    vec3f *dTotalIntegral;
    cudaMalloc(&dTotalIntegral, totalIntegral.size() * sizeof(vec3f));

    int totalMeshLights = meshLightList.size();
    int totalTriLights = triLightList.size();
    cudaMemcpyToSymbol(meshLightSize, &totalMeshLights, sizeof(int));
    cudaMemcpyToSymbol(triLightSize, &totalTriLights, sizeof(int));

    cudaMemcpy(dMeshLights, meshLightList.data(), meshLightList.size() * sizeof(MeshLight), cudaMemcpyHostToDevice);
    cudaMemcpy(dTriLights, triLightList.data(), triLightList.size() * sizeof(TriLight), cudaMemcpyHostToDevice);

    std::cout << "Choosing " << MAX_ELEMS * 3 << " edges" << std::endl;
    std::cout << "Launching with " << blockSize << " blocks with " << threadCount << " threads per block" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    // Launch kernel
    test_ltc_rand<<<blockSize,threadCount>>>(dMeshLights, dTriLights, dTotalIntegral);
    // Wait for kernel to finish executing
    cudaDeviceSynchronize();
    auto finish = std::chrono::high_resolution_clock::now();
    auto milliseconds_taken = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
    std::cout << "Time taken for random triangles: " << milliseconds_taken << "ms" << std::endl;
    // Copy back the chosen lights
    cudaMemcpy(totalIntegral.data(), dTotalIntegral, totalIntegral.size() * sizeof(vec3f), cudaMemcpyDeviceToHost);
    std::cout << "Integral: " << totalIntegral[10].x << " " << totalIntegral[10].y << " " << totalIntegral[10].z << std::endl;

    start = std::chrono::high_resolution_clock::now();
    // Launch kernel
    test_ltc_seq<<<blockSize,threadCount>>>(dMeshLights, dTriLights, dTotalIntegral);
    // Wait for kernel to finish executing
    cudaDeviceSynchronize();
    finish = std::chrono::high_resolution_clock::now();
    milliseconds_taken = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
    std::cout << "Time taken for sequential triangles: " << milliseconds_taken << "ms" << std::endl;
    // Copy back the chosen lights
    cudaMemcpy(totalIntegral.data(), dTotalIntegral, totalIntegral.size() * sizeof(vec3f), cudaMemcpyDeviceToHost);
    std::cout << "Integral: " << totalIntegral[10].x << " " << totalIntegral[10].y << " " << totalIntegral[10].z << std::endl;

    start = std::chrono::high_resolution_clock::now();
    // Launch kernel
    test_ltc_all<<<blockSize,threadCount>>>(dMeshLights, dTriLights, dTotalIntegral);
    // Wait for kernel to finish executing
    cudaDeviceSynchronize();
    finish = std::chrono::high_resolution_clock::now();
    milliseconds_taken = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
    std::cout << "Time taken for all triangles: " << milliseconds_taken << "ms" << std::endl;
    // Copy back the chosen lights
    cudaMemcpy(totalIntegral.data(), dTotalIntegral, totalIntegral.size() * sizeof(vec3f), cudaMemcpyDeviceToHost);
    std::cout << "Integral: " << totalIntegral[10].x << " " << totalIntegral[10].y << " " << totalIntegral[10].z << std::endl;

    cudaFree(dMeshLights);
    cudaFree(dTriLights);
    cudaFree(dTotalIntegral);
}