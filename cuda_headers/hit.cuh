#pragma once

#include "common.cuh"
#include "utils.cuh"
#include <optix_device.h>

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
    // TODO: Fix this to work with multiple bounces
    si.wo = owl::normalize( optixLaunchParams.camera.pos - si.p );
    si.uv = barycentricInterpolate(self.texCoord, primitiveIndices);
    // TODO: Change this to truncf if needed
    si.uv.u = abs(fmodf(si.uv.u, 1.0));
    si.uv.v = abs(fmodf(si.uv.v, 1.0));
    si.n_geom = normalize( barycentricInterpolate(self.normal, primitiveIndices) );
    orthonormalBasis(si.n_geom, si.to_local, si.to_world);

    si.wo_local = normalize(apply_mat(si.to_local, si.wo));

    si.diffuse = self.diffuse;
    if (self.hasDiffuseTexture)
        si.diffuse = (vec3f) tex2D<float4>(self.diffuse_texture, si.uv.x, si.uv.y);

    si.alpha = self.alpha;
    if (self.hasAlphaTexture) {
        vec4f tmp = tex2D<float4>(self.alpha_texture, si.uv.x, si.uv.y);
        si.alpha = length(vec3f(tmp.x, tmp.y, tmp.z));
    }
    si.alpha = owl::clamp(si.alpha, 0.01f, 1.f);

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

