#include "common.cuh"
#include "utils.cuh"

// These headers depend on functions included in common.cuh and utils.cuh
#include "ltc_many_lights_cuda.cuh"
#include "frostbite.cuh"

#include "ltc_utils.cuh"
#include "polygon_utils.cuh"
#include "lcg_random.h"

struct BST {
    int data = -1;
    int left = -1;
    int right = -1;
};

__device__ void stochasticTraverseLBVH(LightBVH* bvh, int bvhHeight, int rootNodeIdx, SurfaceInteraction& si, int& selectedIdx,
    float& lightSelectionPdf, vec2f randVec);
__device__ float deterministicTraverseLBVH(LightBVH* bvh, int bvhHeight, int rootNodeIdx, SurfaceInteraction& si, vec3f point, int& idx);

__device__ void selectFromLBVH(SurfaceInteraction& si, int& selectedIdx, float& lightSelectionPdf, vec2f rand0, vec2f rand1);
__device__ float pdfFromLBVH(SurfaceInteraction& si, vec3f point);

__device__ vec3f integrateOverPolygon(SurfaceInteraction& si, vec3f ltc_mat[3], vec3f ltc_mat_inv[3],
    float amplitude, vec3f iso_frame[3], TriLight& triLight);

__device__ vec3f estimateDirectLighting(SurfaceInteraction& si, LCGRand& rng, int type);
__device__ vec3f estimateDirectLightingLBVH(SurfaceInteraction& si, LCGRand& rng, int type);
__device__ vec3f ltcDirecLighingBaseline(SurfaceInteraction& si, LCGRand& rng);

__device__ vec3f ltcDirectLightingLBVH(SurfaceInteraction& si, LCGRand& rng);


__device__ vec3f sampleLightSource(SurfaceInteraction si, int lightIdx, float lightSelectionPdf, vec2f rand, bool mis);
__device__ vec3f sampleBRDF(SurfaceInteraction si, float lightSelectionPdf, vec2f rand, bool mis);

OPTIX_RAYGEN_PROGRAM(rayGen)()
{
    const RayGenData& self = owl::getProgramData<RayGenData>();
    const vec2i pixelId = owl::getLaunchIndex();
    const int fbOfs = pixelId.x + self.frameBufferSize.x * pixelId.y;

    LCGRand rng = get_rng(optixLaunchParams.accumId, make_uint2(pixelId.x, pixelId.y), 
        make_uint2(self.frameBufferSize.x, self.frameBufferSize.y));

    const vec2f screen = (vec2f(pixelId) + +vec2f(lcg_randomf(rng), lcg_randomf(rng))) / vec2f(self.frameBufferSize);
    RadianceRay ray;
    ray.origin
        = optixLaunchParams.camera.pos;
    ray.direction
        = normalize(optixLaunchParams.camera.dir_00
            + screen.u * optixLaunchParams.camera.dir_du
            + screen.v * optixLaunchParams.camera.dir_dv);

    SurfaceInteraction si;
    owl::traceRay(optixLaunchParams.world, ray, si);

    vec3f color(0.f, 0.f, 0.f);

    if (si.hit == false)
        color = si.diffuse;
    else if (optixLaunchParams.rendererType == DIFFUSE)
        color = si.diffuse;
    else if (optixLaunchParams.rendererType == ALPHA)
        color = si.alpha;
    else if (optixLaunchParams.rendererType == NORMALS)
        color = 0.5f * (si.n_geom + 1.f);
    // Direct lighting with MC
    else if (optixLaunchParams.rendererType == DIRECT_LIGHT_LSAMPLE) {
        if (si.isLight)
            color = si.emit;
        else
            color = estimateDirectLighting(si, rng, 0);
    }
    else if (optixLaunchParams.rendererType == DIRECT_LIGHT_BRDFSAMPLE) {
        if (si.isLight)
            color = si.emit;
        else
            color = estimateDirectLighting(si, rng, 1);
    }
    else if (optixLaunchParams.rendererType == DIRECT_LIGHT_MIS) {
        if (si.isLight)
            color = si.emit;
        else
            color = estimateDirectLighting(si, rng, 2);
    }
    // Direct lighting with MC and LBVH
    else if (optixLaunchParams.rendererType == DIRECT_LIGHT_LBVH_LSAMPLE) {
        if (si.isLight)
            color = si.emit;
        else
            color = estimateDirectLightingLBVH(si, rng, 0);
    }
    else if (optixLaunchParams.rendererType == DIRECT_LIGHT_LBVH_BRDFSAMPLE) {
        if (si.isLight)
            color = si.emit;
        else
            color = estimateDirectLightingLBVH(si, rng, 1);
    }
    else if (optixLaunchParams.rendererType == DIRECT_LIGHT_LBVH_MIS) {
        if (si.isLight)
            color = si.emit;
        else
            color = estimateDirectLightingLBVH(si, rng, 2);
    }
    // Direct lighting with LTC
    else if (optixLaunchParams.rendererType == LTC_BASELINE) {
        if (si.isLight)
            color = si.emit;
        else
            color = ltcDirecLighingBaseline(si, rng);
    }
    else if (optixLaunchParams.rendererType == LTC_LBVH_LINEAR) {
        if (si.isLight)
            color = si.emit;
        else
            color = ltcDirectLightingLBVH(si, rng);
    }
    else if (optixLaunchParams.rendererType == LTC_LBVH_BST) {
        if (si.isLight)
            color = si.emit;
        else
            color = ltcDirectLightingLBVH(si, rng);
    }

    if (optixLaunchParams.accumId > 0)
        color = color + vec3f(optixLaunchParams.accumBuffer[fbOfs]);

    optixLaunchParams.accumBuffer[fbOfs] = vec4f(color, 1.f);
    color = (1.f / (optixLaunchParams.accumId + 1)) * color;
    self.frameBuffer[fbOfs] = owl::make_rgba(color);   
}

OPTIX_CLOSEST_HIT_PROGRAM(triangleMeshCHShadow)()
{
    const TriangleMeshData& self = owl::getProgramData<TriangleMeshData>();
    const vec3i primitiveIndices = self.index[optixGetPrimitiveIndex()];
    ShadowRayData& srd = owl::getPRD<ShadowRayData>();

    if (self.isLight) {
        srd.visibility = vec3f(1.f);
        srd.point = barycentricInterpolate(self.vertex, primitiveIndices);
        srd.normal = normalize(barycentricInterpolate(self.normal, primitiveIndices));
        srd.emit = self.emit;

        vec3f v1 = self.vertex[primitiveIndices.x];
        vec3f v2 = self.vertex[primitiveIndices.y];
        vec3f v3 = self.vertex[primitiveIndices.z];
        srd.area = 0.5f * length(cross(v1 - v2, v3 - v2));

        srd.cg = (v1 + v2 + v3) / 3.f;
    }
    else {
        srd.visibility = vec3f(0.f);
    }

}

OPTIX_CLOSEST_HIT_PROGRAM(triangleMeshCH)()
{
    const TriangleMeshData& self = owl::getProgramData<TriangleMeshData>();
    const vec3i primitiveIndices = self.index[optixGetPrimitiveIndex()];

    SurfaceInteraction& si = owl::getPRD<SurfaceInteraction>();
    si.p = barycentricInterpolate(self.vertex, primitiveIndices);
    si.wo = owl::normalize( optixLaunchParams.camera.pos - si.p );
    si.uv = barycentricInterpolate(self.texCoord, primitiveIndices);
    si.n_geom = normalize( barycentricInterpolate(self.normal, primitiveIndices) );
    orthonormalBasis(si.n_geom, si.to_local, si.to_world);

    si.wo_local = normalize(apply_mat(si.to_local, si.wo));

    si.diffuse = self.diffuse;
    if (self.hasDiffuseTexture)
        si.diffuse = (vec3f) tex2D<float4>(self.diffuse_texture, si.uv.x, si.uv.y);

    si.alpha = self.alpha;
    if (self.hasAlphaTexture)
        si.alpha = tex2D<float4>(self.alpha_texture, si.uv.x, si.uv.y).x;
    si.alpha = clamp(si.alpha, 0.01f, 1.f);

    si.emit = self.emit;
    si.isLight = self.isLight;

    si.hit = true;
}

OPTIX_MISS_PROGRAM(miss)()
{
    const vec2i pixelId = owl::getLaunchIndex();
    const MissProgData& self = owl::getProgramData<MissProgData>();

    SurfaceInteraction& si = owl::getPRD<SurfaceInteraction>();
    si.hit = false;
    si.diffuse = self.const_color;
}

__device__
float deterministicTraverseLBVH(LightBVH* bvh, int bvhHeight, int rootNodeIdx, SurfaceInteraction& si, vec3f point, int& idx)
{
    float pdf = 1.f;

    int nodeIdx = rootNodeIdx;
    for (int i = 0; i < bvhHeight + 1; i++) {
        LightBVH node = bvh[nodeIdx];

        if (node.left == 0 && node.right == 0) {
            idx = node.primIdx;
            if (node.primCount != 1) {
                pdf *= 1.f / node.primCount;
            }
            break;
        }

        LightBVH leftNode = bvh[node.left];
        LightBVH rightNode = bvh[node.right];

        float leftImp = leftNode.flux / pow(owl::length(leftNode.aabbMid - si.p), 2.f);
        float rightImp = rightNode.flux / pow(owl::length(rightNode.aabbMid - si.p), 2.f);
        float sum = leftImp + rightImp;

        leftImp = leftImp / sum;
        rightImp = rightImp / sum;

        if (point.x >= leftNode.aabbMin.x && point.x <= leftNode.aabbMax.x
            && point.y >= leftNode.aabbMin.y && point.y <= leftNode.aabbMax.y
            && point.z >= leftNode.aabbMin.z && point.z <= leftNode.aabbMax.z) {
            nodeIdx = node.left;
            pdf *= leftImp;
        }
        else if (point.x >= rightNode.aabbMin.x && point.x <= rightNode.aabbMax.x
            && point.y >= rightNode.aabbMin.y && point.y <= rightNode.aabbMax.y
            && point.z >= rightNode.aabbMin.z && point.z <= rightNode.aabbMax.z) {
            nodeIdx = node.right;
            pdf *= rightImp;
        }
        else {
            break;
        }
    }

    return pdf;
}

__device__
void stochasticTraverseLBVH(LightBVH* bvh, int bvhHeight, int rootNodeIdx, SurfaceInteraction& si, int& selectedIdx,
                    float& lightSelectionPdf, vec2f randVec)
{
    selectedIdx = -1;
    lightSelectionPdf = 1.f;

    float r1 = randVec.x;
    float r2 = randVec.y;

    int nodeIdx = rootNodeIdx;
    for (int i = 0; i < bvhHeight + 1; i++) {
        LightBVH node = bvh[nodeIdx];

        // If leaf
        if (node.left == 0 && node.right == 0) {
            if (node.primCount == 1) {
                selectedIdx = node.primIdx;
            }
            else {
                selectedIdx = node.primIdx + round(r1 * (node.primCount-1));
                lightSelectionPdf *= 1.f / node.primCount;
            }

            break;
        }

        LightBVH leftNode = bvh[node.left];
        LightBVH rightNode = bvh[node.right];

        float leftImp = leftNode.flux / pow(owl::length(leftNode.aabbMid - si.p), 2.f);
        float rightImp = rightNode.flux / pow(owl::length(rightNode.aabbMid - si.p), 2.f);
        float sum = leftImp + rightImp;

        leftImp = leftImp / sum;
        rightImp = rightImp / sum;

        float eps = r2;
        if (eps < leftImp) {
            nodeIdx = node.left;
            lightSelectionPdf *= leftImp;
        }
        else {
            nodeIdx = node.right;
            lightSelectionPdf *= rightImp;
        }

        if (r1 < leftImp)
            r1 = r1 / leftImp;
        else
            r1 = (r1 - leftImp) / rightImp;

        if (r2 < leftImp)
            r2 = r2 / leftImp;
        else
            r2 = (r2 - leftImp) / rightImp;
    }
}

__device__ 
void selectFromLBVH(SurfaceInteraction& si, int& selectedIdx, float& lightSelectionPdf, vec2f rand0, vec2f rand1)
{
    // First, traverse the light TLAS and retrive the mesh light
    float lightTlasPdf = 1.f;
    int lightTlasIdx = 0;
    int lightTlasRootNodeIdx = 0;

    stochasticTraverseLBVH(optixLaunchParams.lightTlas, optixLaunchParams.lightTlasHeight, lightTlasRootNodeIdx,
        si, lightTlasIdx, lightTlasPdf, rand0);

    MeshLight meshLight = optixLaunchParams.meshLights[lightTlasIdx];

    // Finally, traverse the light BLAS and get the actual triangle
    float lightBlasPdf = 1.f;
    int lightBlasIdx = 0;
    int lightBlasRootNodeIdx = meshLight.bvhIdx;

    stochasticTraverseLBVH(optixLaunchParams.lightBlas, meshLight.bvhHeight, lightBlasRootNodeIdx,
        si, lightBlasIdx, lightBlasPdf, rand1);

    selectedIdx = lightBlasIdx;
    lightSelectionPdf = lightTlasPdf * lightBlasPdf;
}

__device__ 
float pdfFromLBVH(SurfaceInteraction& si, vec3f point)
{
    int meshIdx = 0;
    float tlasPdf = deterministicTraverseLBVH(optixLaunchParams.lightTlas, optixLaunchParams.lightTlasHeight, 
        0, si, point, meshIdx);
    
    MeshLight meshLight = optixLaunchParams.meshLights[meshIdx];
    int triIdx = 0;
    float blasPdf = deterministicTraverseLBVH(optixLaunchParams.lightBlas, meshLight.bvhHeight,
        meshLight.bvhIdx, si, point, triIdx);

    return tlasPdf * blasPdf;
}

__device__
vec3f integrateOverPolygon(SurfaceInteraction& si, vec3f ltc_mat[3], vec3f ltc_mat_inv[3], float amplitude,
    vec3f iso_frame[3], TriLight& triLight)
{
    vec3f lv1 = triLight.v1;
    vec3f lv2 = triLight.v2;
    vec3f lv3 = triLight.v3;
    vec3f lemit = triLight.emit;
    vec3f lnormal = triLight.normal;
    float larea = triLight.area;

    // Move to origin and normalize
    lv1 = owl::normalize(lv1 - si.p);
    lv2 = owl::normalize(lv2 - si.p);
    lv3 = owl::normalize(lv3 - si.p);

    vec3f cg = normalize(lv1 + lv2 + lv3);
    if (owl::dot(-cg, lnormal) < 0.f)
        return vec3f(0.f);

    lv1 = owl::normalize(apply_mat(si.to_local, lv1));
    lv2 = owl::normalize(apply_mat(si.to_local, lv2));
    lv3 = owl::normalize(apply_mat(si.to_local, lv3));

    lv1 = owl::normalize(apply_mat(iso_frame, lv1));
    lv2 = owl::normalize(apply_mat(iso_frame, lv2));
    lv3 = owl::normalize(apply_mat(iso_frame, lv3));

    float diffuse_shading = 0.f;
    float ggx_shading = 0.f;

    vec3f diff_clipped[5] = { lv1, lv2, lv3, lv1, lv1 };
    int diff_vcount = clipPolygon(3, diff_clipped);
    
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
    int ltc_vcount = clipPolygon(diff_vcount, ltc_clipped);

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

    vec3f color = (si.diffuse * lemit * diffuse_shading) + (amplitude * lemit * ggx_shading);
    return color;
}

__device__ 
vec3f ltcDirectLightingLBVH(SurfaceInteraction& si, LCGRand& rng)
{
    vec3f normal_local(0.f, 0.f, 1.f);

    vec2f rand0(lcg_randomf(rng), lcg_randomf(rng));
    vec2f rand1(lcg_randomf(rng), lcg_randomf(rng));

    if (si.wo_local.z < 0.f)
        return vec3f(0.f);

    /* Analytic shading via LTCs */
    vec3f ltc_mat[3], ltc_mat_inv[3];
    float alpha = si.alpha;
    float theta = sphericalTheta(si.wo_local);

    float amplitude = 1.f;
    fetchLtcMat(alpha, theta, ltc_mat, amplitude);
    matrixInverse(ltc_mat, ltc_mat_inv);

    vec3f iso_frame[3];

    iso_frame[0] = si.wo_local;
    iso_frame[0].z = 0.f;
    iso_frame[0] = normalize(iso_frame[0]);
    iso_frame[2] = normal_local;
    iso_frame[1] = normalize(owl::cross(iso_frame[2], iso_frame[0]));

    int selectedIdx[MAX_LTC_LIGHTS * 2] = { -1 };
    int selectedEnd = 0;

    int ridx = 0;
    float rpdf = 0.f;
    selectFromLBVH(si, ridx, rpdf, rand0, rand1);

    selectedIdx[selectedEnd++] = ridx;

    for (int i = 0; i < MAX_LTC_LIGHTS*2; i++) {
        if (selectedEnd == optixLaunchParams.numTriLights)
            break;

        rand0 = vec2f(lcg_randomf(rng), lcg_randomf(rng));
        rand1 = vec2f(lcg_randomf(rng), lcg_randomf(rng));

        ridx = 0;
        rpdf = 0.f;
        selectFromLBVH(si, ridx, rpdf, rand0, rand1);

        bool found = false;
        for (int j = 0; j < selectedEnd; j++) {
            if (selectedIdx[j] == ridx) {
                found = true;
                break;
            }
        }

        if (!found) {
            selectedIdx[selectedEnd++] = ridx;
        }
    }

    vec3f color(0.f, 0.f, 0.f);
    for (int i = 0; i < selectedEnd; i++) {
        color += integrateOverPolygon(si, ltc_mat, ltc_mat_inv, amplitude, iso_frame,
            optixLaunchParams.triLights[selectedIdx[i]]);
    }

    return color;
}

// __device__ 
// vec3f ltcDirectLightingLBVH(SurfaceInteraction& si, bool useBst)
// {
//     const vec2i pixelId = owl::getLaunchIndex();
//     owl::common::LCG<MAX_LTC_LIGHTS*4> rng(pixelId.x * pixelId.y, optixLaunchParams.accumId);
//     float eps1 = rng();
//     float eps2 = rng();
// 
//     vec3f wo_local = normalize(apply_mat(si.to_local, si.wo));
//     if (wo_local.z < 0.f)
//         return vec3f(0.f);
// 
//     vec3f normal_local(0.f, 0.f, 1.f);
//     vec3f color(0.0, 0.0, 0.0);
// 
//     /* Analytic shading via LTCs */
//     vec3f ltc_mat[3], ltc_mat_inv[3];
//     float alpha = si.alpha;
//     float theta = sphericalTheta(wo_local);
// 
//     float amplitude = 1.f;
//     fetchLtcMat(alpha, theta, ltc_mat, amplitude);
//     matrixInverse(ltc_mat, ltc_mat_inv);
// 
//     vec3f iso_frame[3];
// 
//     iso_frame[0] = wo_local;
//     iso_frame[0].z = 0.f;
//     iso_frame[0] = normalize(iso_frame[0]);
//     iso_frame[2] = normal_local;
//     iso_frame[1] = normalize(owl::cross(iso_frame[2], iso_frame[0]));
// 
//     int selectedIdx[MAX_LTC_LIGHTS * 2] = { -1 };
//     int selectedEnd = 0;
// 
//     if (useBst) {
//         BST set[MAX_LTC_LIGHTS * 2];
//         int setEnd = 0;
// 
//         int numTriLights = optixLaunchParams.numTriLights;
// 
//         int ridx = -1;
//         float rpdf = 1.f;
//         traverseLBVH(si, ridx, rpdf, vec2f(rng(), rng()));
// 
//         set[setEnd++].data = ridx;
//         selectedIdx[selectedEnd++] = ridx;
// 
//         [[ unroll ]]
//         for (int i = 0; i < MAX_LTC_LIGHTS; i++) {
//             ridx = -1;
//             rpdf = 1.f;
//             traverseLBVH(si, ridx, rpdf, vec2f(rng(), rng()));
// 
//             int setIdx = 0;
//             bool found = false;
//             [[ unroll ]]
//             for (int j = 0; j < MAX_LTC_LIGHTS; j++) {
// 
//                 // If found
//                 if (set[setIdx].data == ridx) {
//                     found = true;
//                     break;
//                 }
// 
//                 // Insert if empty node
//                 if (set[setIdx].data == -1 && set[setIdx].left == -1 && set[setIdx].right == -1) {
//                     set[setIdx].data = ridx;
//                     break;
//                 }
// 
//                 // If child
//                 if (set[setIdx].data != -1 && set[setIdx].left == -1 && set[setIdx].right == -1) {
//                     set[setEnd++].data = ridx;
//                     set[setEnd++].data = -1;
// 
//                     if (ridx > set[setIdx].data) {
//                         set[setIdx].right = setEnd - 2;
//                         set[setIdx].left = setEnd - 1;
//                     }
//                     else {
//                         set[setIdx].right = setEnd - 1;
//                         set[setIdx].left = setEnd - 2;
//                     }
// 
//                     break;
//                 }
// 
//                 if (ridx > set[setIdx].data) {
//                     setIdx = set[setIdx].right;
//                 }
//                 else {
//                     setIdx = set[setIdx].left;
//                 }
// 
//             }
// 
//             if (!found)
//                 selectedIdx[selectedEnd++] = ridx;
//         }
//     }
//     else {
//         int numTriLights = optixLaunchParams.numTriLights;
// 
//         int ridx = -1;
//         float rpdf = 1.f;
//         traverseLBVH(si, ridx, rpdf, vec2f(rng(), rng()));
//         selectedIdx[selectedEnd++] = ridx;
// 
//         for (int i = 0; i < MAX_LTC_LIGHTS*2; i++) {
//             traverseLBVH(si, ridx, rpdf, vec2f(rng(), rng()));
// 
//             bool found = false;
//             for (int j = 0; j < selectedEnd; j++) {
//                 if (selectedIdx[j] == ridx) {
//                     found = true;
//                     break;
//                 }
//             }
// 
//             if (!found) {
//                 selectedIdx[selectedEnd++] = ridx;
//             }
//         }
//     }
// 
//     [[ unroll ]]
//     for (int i = 0; i < selectedEnd; i++) {
//         color += integrateOverPolygon(si, ltc_mat, ltc_mat_inv, amplitude, iso_frame,
//             optixLaunchParams.triLights[selectedIdx[i]]);
//     }
// 
//     return color;
// }

__device__
vec3f ltcDirecLighingBaseline(SurfaceInteraction& si, LCGRand& rng)
{
    vec3f wo_local = normalize(apply_mat(si.to_local, si.wo));
    if (wo_local.z < 0.f)
        return vec3f(0.f);

    vec3f normal_local(0.f, 0.f, 1.f);
    vec3f color(0.0, 0.0, 0.0);

    /* Analytic shading via LTCs */
    vec3f ltc_mat[3], ltc_mat_inv[3];
    float alpha = si.alpha;
    float theta = sphericalTheta(wo_local);

    float amplitude = 1.f;
    fetchLtcMat(alpha, theta, ltc_mat, amplitude);
    matrixInverse(ltc_mat, ltc_mat_inv);

    vec3f iso_frame[3];

    iso_frame[0] = wo_local;
    iso_frame[0].z = 0.f;
    iso_frame[0] = normalize(iso_frame[0]);
    iso_frame[2] = normal_local;
    iso_frame[1] = normalize(owl::cross(iso_frame[2], iso_frame[0]));

    for (int lidx = 0; lidx < optixLaunchParams.numTriLights; lidx++) {
        color += integrateOverPolygon(si, ltc_mat, ltc_mat_inv, amplitude, iso_frame, 
                                            optixLaunchParams.triLights[lidx]);
    }

    return color;
}

__device__
vec3f estimateDirectLightingLBVH(SurfaceInteraction& si, LCGRand& rng, int type)
{
    vec2f rand0(lcg_randomf(rng), lcg_randomf(rng));
    vec2f rand1(lcg_randomf(rng), lcg_randomf(rng));
    vec2f rand2(lcg_randomf(rng), lcg_randomf(rng));
    vec2f rand3(lcg_randomf(rng), lcg_randomf(rng));

    int selectedTriLight = 0;
    float lightSelectionPdf = 0.f;
    selectFromLBVH(si, selectedTriLight, lightSelectionPdf, rand0, rand1);

    vec3f lightSample = vec3f(0.f);
    vec3f brdfSample = vec3f(0.f);

    if (type == 0) {
        lightSample = sampleLightSource(si, selectedTriLight, lightSelectionPdf, rand2, false);
    }
    else if (type == 1) {
        brdfSample = sampleBRDF(si, lightSelectionPdf, rand3, false);
    }
    else if (type == 2) {
        brdfSample = sampleBRDF(si, lightSelectionPdf, rand2, true);
        lightSample = sampleLightSource(si, selectedTriLight, lightSelectionPdf, rand3, true);
    }

    // Make sure there are no negative colors!
    vec3f color = lightSample + brdfSample;
    color.x = owl::max(0.f, color.x);
    color.y = owl::max(0.f, color.y);
    color.z = owl::max(0.f, color.z);

    return color;
}

__device__
vec3f estimateDirectLighting(SurfaceInteraction& si, LCGRand& rng, int type)
{
    vec2f rand1 = vec2f(lcg_randomf(rng), lcg_randomf(rng));
    vec2f rand2 = vec2f(lcg_randomf(rng), lcg_randomf(rng));

    int selectedTriLight = round(lcg_randomf(rng) * (optixLaunchParams.numTriLights-1));
    float lightSelectionPdf = 1.f / optixLaunchParams.numTriLights;

    vec3f lightSample = vec3f(0.f);
    vec3f brdfSample = vec3f(0.f);

    if (type == 0) {
        lightSample = sampleLightSource(si, selectedTriLight, lightSelectionPdf, rand1, false);
    }
    else if (type == 1) {
        brdfSample = sampleBRDF(si, lightSelectionPdf, rand2, false);
    }
    else if (type == 2) {
        brdfSample = sampleBRDF(si, lightSelectionPdf, rand1, true);
        lightSample = sampleLightSource(si, selectedTriLight, lightSelectionPdf, rand2, true);
    }

    // Make sure there are no negative colors!
    vec3f color = lightSample + brdfSample;
    color.x = owl::max(0.f, color.x);
    color.y = owl::max(0.f, color.y);
    color.z = owl::max(0.f, color.z);

    return color;
}

__device__
vec3f sampleLightSource(SurfaceInteraction si, int lightIdx, float lightSelectionPdf, vec2f rand, bool mis)
{
    vec3f color(0.f, 0.f, 0.f);
    float light_pdf = 0.f, brdf_pdf = 0.f;
    TriLight triLight = optixLaunchParams.triLights[lightIdx];

    vec3f lv1 = triLight.v1;
    vec3f lv2 = triLight.v2;
    vec3f lv3 = triLight.v3;
    vec3f lnormal = triLight.normal;
    vec3f lemit = triLight.emit;
    float larea = triLight.area;

    vec3f lpoint = samplePointOnTriangle(lv1, lv2, lv3, rand.x, rand.y);
    si.wi = normalize(lpoint - si.p);
    si.wi_local = normalize(apply_mat(si.to_local, si.wi));

    float xmy = pow(owl::length(lpoint - si.p), 2.f);
    float lDotWi = owl::abs(owl::dot(lnormal, -si.wi));

    light_pdf = lightSelectionPdf * (xmy / (larea * lDotWi));

    ShadowRay ray;
    ray.origin = si.p + 1e-3f * si.n_geom;
    ray.direction = si.wi;

    ShadowRayData srd;
    owl::traceRay(optixLaunchParams.world, ray, srd);

    if (si.wo_local.z > 0.f && si.wi_local.z > 0.f && srd.visibility != vec3f(0.f) && light_pdf > 0.f && owl::dot(-si.wi, lnormal) > 0.f) {
        vec3f brdf = evaluate_brdf(si.wo_local, si.wi_local, si.diffuse, si.alpha);
        brdf_pdf = get_brdf_pdf(si.alpha, si.wo_local, normalize(si.wo_local + si.wi_local));

        if (mis && brdf_pdf > 0.f) {
            float weight = PowerHeuristic(1, light_pdf, 1, brdf_pdf);
            color += brdf * lemit * owl::abs(si.wi_local.z) * weight / light_pdf;
        }
        else if(!mis) {
            color += brdf * lemit * owl::abs(si.wi_local.z) / light_pdf;
        }
    }

    return color;
}

__device__
vec3f sampleBRDF(SurfaceInteraction si, float lightSelectionPdf, vec2f rand, bool mis)
{
    si.wi_local = sample_GGX(rand, si.alpha, si.wo_local);
    si.wi = normalize(apply_mat(si.to_world, si.wi_local));

    ShadowRay ray;
    ShadowRayData srd;
    ray.origin = si.p + 1e-3f * si.n_geom;
    ray.direction = si.wi;
    owl::traceRay(optixLaunchParams.world, ray, srd);

    vec3f color(0.f, 0.f, 0.f);
    float light_pdf = 0.f, brdf_pdf = 0.f;

    if (si.wi_local.z > 0.f && si.wo_local.z > 0.f && srd.visibility != vec3f(0.f)) {
        float xmy = pow(owl::length(srd.point - si.p), 2.f);
        float lDotWi = owl::abs(owl::dot(srd.normal, -si.wi));
        light_pdf = lightSelectionPdf * (xmy / (srd.area * lDotWi));

        vec3f brdf = evaluate_brdf(si.wo_local, si.wi_local, si.diffuse, si.alpha);
        brdf_pdf = get_brdf_pdf(si.alpha, si.wo_local, normalize(si.wo_local + si.wi_local));

        if (mis && light_pdf > 0.f && brdf_pdf > 0.f) {
            float weight = PowerHeuristic(1, brdf_pdf, 1, light_pdf);
            color += brdf * srd.emit * owl::abs(si.wi_local.z) * weight / brdf_pdf;
        }
        else if (!mis && brdf_pdf > 0.f) {
            color += brdf * srd.emit * owl::abs(si.wi_local.z) / brdf_pdf;
        }
    }

    return color;
}

