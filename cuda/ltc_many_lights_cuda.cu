#include "common.cuh"
#include "utils.cuh"

// These headers depend on functions included in common.cuh and utils.cuh
#include "ltc_many_lights_cuda.cuh"
#include "frostbite.cuh"

static __device__ vec3f estimateDirectLighting(SurfaceInteraction& si);

OPTIX_RAYGEN_PROGRAM(rayGen)()
{
    const RayGenData& self = owl::getProgramData<RayGenData>();
    const vec2i pixelId = owl::getLaunchIndex();
    const int fbOfs = pixelId.x + self.frameBufferSize.x * pixelId.y;

    owl::common::LCG<4> rng(pixelId.x*pixelId.y, optixLaunchParams.accumId);

    const vec2f screen = (vec2f(pixelId) + +vec2f(rng(), rng())) / vec2f(self.frameBufferSize);
    RadianceRay ray;
    ray.origin
        = optixLaunchParams.camera.pos;
    ray.direction
        = normalize(optixLaunchParams.camera.dir_00
            + screen.u * optixLaunchParams.camera.dir_du
            + screen.v * optixLaunchParams.camera.dir_dv);

    vec3f color;
    owl::traceRay(/*accel to trace against*/optixLaunchParams.world,
        /*the ray to trace*/ray,
        /*prd*/color);

    if (optixLaunchParams.accumId > 0)
        color = color + vec3f(optixLaunchParams.accumBuffer[fbOfs]);

    optixLaunchParams.accumBuffer[fbOfs] = vec4f(color, 1.f);
    color = (1.f / (optixLaunchParams.accumId + 1)) * color;
    self.frameBuffer[fbOfs] = owl::make_rgba(color);
    
}

OPTIX_ANY_HIT_PROGRAM(triangleMeshAH)()
{
    const TriangleMeshData& self = owl::getProgramData<TriangleMeshData>();
    vec3f& visibility = owl::getPRD<vec3f>();

    if (self.isLight)
        visibility = vec3f(1.f);
    else
        visibility = vec3f(0.f);
}

OPTIX_CLOSEST_HIT_PROGRAM(triangleMeshCH)()
{
    const TriangleMeshData& self = owl::getProgramData<TriangleMeshData>();
    vec3f& color = owl::getPRD<vec3f>();
    const vec3i primitiveIndices = self.index[optixGetPrimitiveIndex()];

    SurfaceInteraction si;
    si.p = barycentricInterpolate(self.vertex, primitiveIndices);
    si.wo = owl::normalize( optixLaunchParams.camera.pos - si.p );
    si.uv = barycentricInterpolate(self.texCoord, primitiveIndices);
    si.n_geom = normalize( barycentricInterpolate(self.normal, primitiveIndices) );
    orthonormalBasis(si.n_geom, si.to_local, si.to_world);

    si.diffuse = self.diffuse;
    if (self.hasDiffuseTexture)
        si.diffuse = (vec3f) tex2D<float4>(self.diffuse_texture, si.uv.x, si.uv.y);

    si.alpha = self.alpha;
    if (self.hasAlphaTexture)
        si.alpha = tex2D<float4>(self.alpha_texture, si.uv.x, si.uv.y).x;
    si.alpha = clamp(si.alpha, 0.01f, 1.f);

    if (self.isLight)
        color = self.emit;
    else
        color = estimateDirectLighting(si);
}

OPTIX_MISS_PROGRAM(miss)()
{
    const vec2i pixelId = owl::getLaunchIndex();

    const MissProgData& self = owl::getProgramData<MissProgData>();

    vec3f& prd = owl::getPRD<vec3f>();
    prd = self.const_color;
}

static __device__
vec3f estimateDirectLighting(SurfaceInteraction& si)
{
    const vec2i pixelId = owl::getLaunchIndex();
    owl::common::LCG<4> rng(pixelId.x * pixelId.y, optixLaunchParams.accumId);

    vec3f color(0.f, 0.f, 0.f);
    vec3f wo_local = normalize( apply_mat(si.to_local, si.wo) );
    vec3f normal_local(0.f, 0.f, 1.f);

    int selectedAreaLight = round(rng() * optixLaunchParams.numAreaLights);
    TriLight areaLight = optixLaunchParams.areaLights[selectedAreaLight];

    {
        vec3f lv1 = areaLight.v1;
        vec3f lv2 = areaLight.v2;
        vec3f lv3 = areaLight.v3;
        vec3f lnormal = areaLight.normal;
        vec3f lemit = areaLight.emissionRadiance;
        float larea = areaLight.area;

        vec3f lpoint = samplePointOnTriangle(lv1, lv2, lv3, rng(), rng());
        vec3f wi = normalize(lpoint - si.p);
        vec3f wi_local = normalize( apply_mat(si.to_local, wi) );

        float xmy = pow(owl::length(lpoint - si.p), 2.f);
        float lDotWi = owl::abs(owl::dot(lnormal, -wi));

        float pdf = (1.f / optixLaunchParams.numAreaLights) * (xmy / (larea * lDotWi));

        ShadowRay ray;
        ray.origin = si.p + 1e-3f * si.n_geom;
        ray.direction = wi;

        vec3f visibility;
        owl::traceRay(optixLaunchParams.world, ray, visibility);

        if (wo_local.z > 0.f && wi_local.z > 0.f) {
            vec3f brdf = evaluate_brdf(wo_local, wi_local, si.diffuse, si.alpha);
            color += brdf * lemit * visibility * owl::abs(wi_local.z) / pdf;
        }
    }

    return color;
}

