#include "interactive.cuh"
#include <optix_device.h>

OPTIX_RAYGEN_PROGRAM(rayGen)()
{
    const RayGenData& self = owl::getProgramData<RayGenData>();
    const vec2i pixelID = owl::getLaunchIndex();
    const int fbOfs = pixelID.x + self.frameBufferSize.x * pixelID.y;

    Random rng(fbOfs, optixLaunchParams.accumId);

    const vec2f screen = (vec2f(pixelID) + +vec2f(rng(), rng())) / vec2f(self.frameBufferSize);
    owl::Ray ray;
    ray.origin
        = self.camera.pos;
    ray.direction
        = normalize(self.camera.dir_00
            + screen.u * self.camera.dir_du
            + screen.v * self.camera.dir_dv);

    vec3f color;
    owl::traceRay(/*accel to trace against*/self.world,
        /*the ray to trace*/ray,
        /*prd*/color);

    if (optixLaunchParams.accumId > 0)
        color = color + vec3f(optixLaunchParams.accumBuffer[fbOfs]);

    optixLaunchParams.accumBuffer[fbOfs] = vec4f(color, 1.f);
    color = (1.f / (optixLaunchParams.accumId + 1)) * color;
    self.frameBuffer[fbOfs] = owl::make_rgba(color);
    
}

OPTIX_CLOSEST_HIT_PROGRAM(triangleMeshCH)()
{
    const TriangleMeshData& self = owl::getProgramData<TriangleMeshData>();
    vec3f& prd = owl::getPRD<vec3f>();
}

OPTIX_MISS_PROGRAM(miss)()
{
    const vec2i pixelID = owl::getLaunchIndex();

    const MissProgData& self = owl::getProgramData<MissProgData>();

    vec3f& prd = owl::getPRD<vec3f>();
    prd = self.const_color;
}

