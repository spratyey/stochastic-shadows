#include "common.cuh"
#include "utils.cuh"

#include "hit.cuh"
#include "renderers/ltc_lbvh_sil.cuh"
#include "renderers/sil_test.cuh"
#include "constants.cuh"

#include "lcg_random.cuh"
#include "constants.cuh"
#include "owl/common/math/vec.h"

OPTIX_RAYGEN_PROGRAM(rayGen)() {
    const RayGenData& self = owl::getProgramData<RayGenData>();
    const vec2i pixelId = owl::getLaunchIndex();
    const int fbOfs = pixelId.x + self.frameBufferSize.x * pixelId.y;

    LCGRand rng = get_rng(optixLaunchParams.accumId + 10007, make_uint2(pixelId.x, pixelId.y),
        make_uint2(self.frameBufferSize.x, self.frameBufferSize.y));

    const vec2f screen = (vec2f(pixelId) + vec2f(lcg_randomf(rng), lcg_randomf(rng))) / vec2f(self.frameBufferSize);
    RadianceRay ray;
    ray.origin
        = optixLaunchParams.camera.pos;
    ray.direction
        = normalize(optixLaunchParams.camera.dir_00
            + screen.u * optixLaunchParams.camera.dir_du
            + screen.v * optixLaunchParams.camera.dir_dv);

    SurfaceInteraction si;
    owl::traceRay(optixLaunchParams.world, ray, si);

    vec3f color(0.f);

    if (si.hit == false) {
        color = si.diffuse;
    } else {
#ifdef DEBUG_SIL
        if (si.isLight) {
            color = colorEdges(si, ray);
        } else {
            color = (vec3f(1) + si.n_geom) / vec3f(2);
        }
        // if (pixelId.x > 374 && pixelId.y < 550) {
        //     color = vec3f(1,0, 0);
        // }
#else
        if (si.isLight) {
            color = si.emit;
        } else {
            color = ltcDirectLightingLBVHSil(si, rng);
        }
#endif
    }

    self.frameBuffer[fbOfs] = owl::make_rgba(color);
}