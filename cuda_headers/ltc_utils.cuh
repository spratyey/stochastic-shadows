#pragma once

#include "owl/common/math/vec.h"
#include "common.cuh"
#include "utils.cuh"
#include "polygon_utils.cuh"
#include "sil_utils.cuh"
#include "types.hpp"

using namespace std;

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
vec3f integrateOverPolygon(SurfaceInteraction& si, vec3f ltc_mat[3], vec3f ltc_mat_inv[3], float amplitude,
    vec3f iso_frame[3], TriLight& triLight)
{
    vec3f lv1 = triLight.v1;
    vec3f lv2 = triLight.v2;
    vec3f lv3 = triLight.v3;
    vec3f lemit = triLight.emit;
    vec3f lnormal = triLight.normal;

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
vec3f integrateOverSil(SurfaceInteraction& si, vec3f mat[3], vec3f mat2[3], BSPNode node, int selectedLightIdx) {
    MeshLight meshLight = optixLaunchParams.meshLights[selectedLightIdx];
    vec3f color(0);
    vec2i silSpan = node.silSpan;

    int edgeCount = meshLight.spans.edgeSpan.y - meshLight.spans.edgeSpan.x;
    
    // Offset the arrays for ease
    int *silhouettes = &optixLaunchParams.silhouettes[meshLight.spans.silSpan.x];
    LightEdge *edges = &optixLaunchParams.lightEdges[meshLight.spans.edgeSpan.x];

    int silIdx = silhouettes[silSpan.x];
    int edgeIdx = abs(silIdx) % edgeCount;
    bool toFlip = silIdx < 0 || silIdx == edgeCount;
    vec3f prevVertex = toFlip ? edges[edgeIdx].v2 : edges[edgeIdx].v1;
    vec3f currVertex;
    vec3f lv1;
    vec3f lv2;
    vec2f clippingSum = vec2f(1, 0);
    vec3f integral = vec3f(0);
    for (int i = silSpan.x; i < silSpan.y; i += 1) {
        silIdx = silhouettes[i];
        edgeIdx = abs(silIdx) % edgeCount;
        toFlip = silIdx < 0 || silIdx == edgeCount;
        currVertex = toFlip ? edges[edgeIdx].v1 : edges[edgeIdx].v2;

        // Move to origin and normalize
        lv1 = normalize(prevVertex - si.p);
        lv2 = normalize(currVertex - si.p);

        // Project to upper sphere
        lv1 = normalize(apply_mat(si.to_local, lv1));
        lv2 = normalize(apply_mat(si.to_local, lv2));

        lv1 = normalize(apply_mat(mat, lv1));
        lv2 = normalize(apply_mat(mat, lv2));

        lv1 = normalize(apply_mat(mat2, lv1));
        lv2 = normalize(apply_mat(mat2, lv2));

        // Clip to upper hemisphere
        bool lv1BelowEquator = lv1.z < 0;
        bool lv2BelowEquator = lv2.z < 0;
        bool bothBelowEquator = lv1BelowEquator && lv2BelowEquator;

        if (lv1BelowEquator != lv2BelowEquator) {
            vec3f p = equatorInteresection(lv1, lv2);

            if (lv1BelowEquator) {
                lv1 = p;
                clippingSum = atan2Sum(clippingSum, vec2f(p.x, p.y));
            } else {
                lv2 = p;
                clippingSum = atan2Diff(clippingSum, vec2f(p.x, p.y));
            }
        }

        // Integrate
        if (!bothBelowEquator) {
            integral += integrateEdgeSil(lv1, lv2);
        }

        prevVertex = currVertex;
    }
    float angle = atan2(clippingSum.y, clippingSum.x);

    angle = fmodf(angle + 2*PI, 2*PI);
    if (abs(2*PI - angle) < 1e-3) {
        angle = 0.0f;
    }
    color = (integral + angle) / (2*PI);

    return color;
}

__device__
vec3f integrateOverPolyhedron(SurfaceInteraction& si, vec3f ltc_mat[3], vec3f ltc_mat_inv[3], float amplitude, vec3f iso_frame[3], BSPNode node, int selectedLightIdx)
{
    vec3f diffuseShading(0, 0, 0);
    vec3f ggxShading(0, 0, 0);
    vec3f identity[3]; 
    identity[0] = vec3f(1, 0, 0);
    identity[1] = vec3f(0, 1, 0);
    identity[2] = vec3f(0, 0, 1);
    MeshLight meshLight = optixLaunchParams.meshLights[selectedLightIdx];
#ifdef BSP_SIL
    diffuseShading = integrateOverSil(si, identity, iso_frame, node, selectedLightIdx);
    ggxShading = integrateOverSil(si, iso_frame, ltc_mat_inv, node, selectedLightIdx);
    // ggxShading = vec3f(0);
#else
    int edgeStartIdx = meshLight.spans.edgeSpan.x;
    int edgeEndIdx = meshLight.spans.edgeSpan.y;
    for (int i = edgeStartIdx; i < edgeEndIdx; i += 1) {
        LightEdge edge = optixLaunchParams.lightEdges[i];
        vec3f n1 = edge.n1;
        bool isSil;
        if (edge.adjFaceCount == 2) {
            vec3f n2 = edge.n2;
            isSil = owl::dot(n1, edge.v1 - si.p) * owl::dot(n2, edge.v2 - si.p) < 0;
        } else {
            isSil = true;
        }

        if (isSil) {
            bool toFlip = shouldFlip(edge, si.p);
            vec3f lv1 = toFlip ? edge.v2 : edge.v1;
            vec3f lv2 = toFlip ? edge.v1 : edge.v2;

            // Move to origin and normalize
            lv1 = owl::normalize(lv1 - si.p);
            lv2 = owl::normalize(lv2 - si.p);

            // Project to upper sphere
            lv1 = owl::normalize(apply_mat(si.to_local, lv1));
            lv2 = owl::normalize(apply_mat(si.to_local, lv2));

            lv1 = owl::normalize(apply_mat(iso_frame, lv1));
            lv2 = owl::normalize(apply_mat(iso_frame, lv2));

            // TODO: Clipping

            diffuseShading += integrateEdge(lv1, lv2);

            lv1 = owl::normalize(apply_mat(ltc_mat_inv, lv1));
            lv2 = owl::normalize(apply_mat(ltc_mat_inv, lv2));

            ggxShading += integrateEdge(lv1, lv2);
        }
    }
#endif
    vec3f color = (si.diffuse * meshLight.avgEmit * diffuseShading) + (amplitude * meshLight.avgEmit * ggxShading);
    return color;
}


__device__
void fetchLtcMat(float alpha, float theta, vec3f ltc_mat[3], float &amplitude)
{
    theta = theta * 0.99f / (0.5 * PI);

    float4 r1 = tex2D<float4>(optixLaunchParams.ltc_1, theta, alpha);
    float4 r2 = tex2D<float4>(optixLaunchParams.ltc_2, theta, alpha);
    float4 r3 = tex2D<float4>(optixLaunchParams.ltc_3, theta, alpha);

    ltc_mat[0] = vec3f(r1.x, r1.y, r1.z);
    ltc_mat[1] = vec3f(r2.x, r2.y, r2.z);
    ltc_mat[2] = vec3f(r3.x, r3.y, r3.z);

    amplitude = r3.w;
}
