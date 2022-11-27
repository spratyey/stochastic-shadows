#include "common.cuh"
#include "utils.cuh"

#include "hit.cuh"
#include "constants.cuh"
#include "renderers/sil_test.cuh"
#include "renderers/ltc_baseline.cuh"
#include "renderers/ltc_lbvh.cuh"
#include "renderers/ltc_lbvh_sil.cuh"
#include "renderers/direct_lighting.cuh"

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

    print_pixel("%d %d %d %d\n", self.frameBufferSize.x, self.frameBufferSize.y, pixelId.x, pixelId.y);

    vec3f color(0.f);

    if (si.hit == false) {
        color = si.diffuse;
    } else {
#if RENDERER == DEBUG_DIFFUSE
        color = si.diffuse;
#elif RENDERER == DEBUG_SIL 
        if (si.isLight) {
            color = colorEdges(si, ray);
        } else {
            color = (vec3f(1) + si.n_geom) / vec3f(2);
        }
#elif RENDERER == LTC_BASE
        if (si.isLight) {
            color = si.emit;
        } else {
            color = ltcDirectLightingBaseline(si, rng);
        }
#elif RENDERER == LTC_SAMPLE_TRI
        if (si.isLight) {
            color = si.emit;
        } else {
            color = ltcDirectLightingLBVH(si, rng);
        }
#elif RENDERER == LTC_SAMPLE_POLY
        if (si.isLight) {
            color = si.emit;
        } else {
            color = ltcDirectLightingLBVHSil(si, rng);
        }
#elif RENDERER == DIRECT_LIGHTING
        if (si.isLight) {
            color = si.emit;
        } else {
            color = estimateDirectLighting(si, rng, 2);
        }
#endif
    }

    self.frameBuffer[fbOfs] = owl::make_rgba(vec3f(linear_to_srgb(color.x), linear_to_srgb(color.y), linear_to_srgb(color.z)));
}