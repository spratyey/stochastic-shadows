#include "common.cuh"
#include "utils.cuh"

#include "hit.cuh"
#include "constants.cuh"
#include "renderers/sil_test.cuh"
#include "renderers/ltc_baseline.cuh"
#include "renderers/ltc_lbvh_poly.cuh"
#include "renderers/ltc_lbvh_tri.cuh"
#include "renderers/ltc_monte_carlo.cuh"
#include "renderers/direct_lighting.cuh"
#include "renderers/restir.cuh"

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

    int binIdx = lcg_randomf(rng)*NUM_BINS;
    
    if (si.hit == false) {
        color = si.diffuse;
    } else {
#if RENDERER == DEBUG_DIFFUSE
        color = si.diffuse;
#elif RENDERER == DEBUG_NORMAL
        color = (si.n_geom + 1) / 2;
#elif RENDERER == DEBUG_ALPHA
        color = vec3f(si.alpha);
        print_pixel("%f\n", si.alpha);
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
            color = ltcDirectLightingBaseline(si, rng, binIdx);
        }
#elif RENDERER == LTC_SAMPLE_TRI
        if (si.isLight) {
            color = si.emit;
        } else {
            color = ltcDirectLightingLBVHTri(si, rng);
        }
#elif RENDERER == LTC_SAMPLE_POLY
        if (si.isLight) {
            color = si.emit;
        } else {
            color = ltcDirectLightingLBVHPoly(si, rng, binIdx);
        }
#elif RENDERER == LTC_MONTE_CARLO
        if (si.isLight) {
            color = si.emit;
        } else {
            color = ltcMonteCarlo(si, rng);
        }
#elif RENDERER == DIRECT_LIGHTING
        if (si.isLight) {
            color = si.emit;
        } else {
            color = estimateDirectLighting(si, rng, 2);
        }
#elif RENDERER == DIRECT_LIGHTING_RESTIR
        if (si.isLight) {
            color = si.emit;
        } else {
            color = estimateDirectLightingReSTIR(si, rng);
        }
#endif
    }

#ifdef ACCUM
    if (optixLaunchParams.accumId > 0) {
        color = color + vec3f(optixLaunchParams.accumBuffer[fbOfs]);
    } 
    optixLaunchParams.accumBuffer[fbOfs] = vec4f(color, 1.f);
    color = (1.f / (optixLaunchParams.accumId + 1)) * color;
#endif

#ifdef SPATIAL_REUSE
    optixLaunchParams.binIdxBuffer[fbOfs] = binIdx;
    optixLaunchParams.normalBuffer[fbOfs] = si.n_geom;
    optixLaunchParams.albedoBuffer[fbOfs] = si.diffuse;
    if (si.hit) {
        optixLaunchParams.depthBuffer[fbOfs] = length(si.p - ray.origin);
    } else {
        optixLaunchParams.depthBuffer[fbOfs] = 0.0;
    }
#endif
    self.frameBuffer[fbOfs] = owl::make_rgba(vec3f(
        linear_to_srgb(color.x),
        linear_to_srgb(color.y),
        linear_to_srgb(color.z)
    ));
}

#ifdef SPATIAL_REUSE
OPTIX_RAYGEN_PROGRAM(spatialReuse)() {
    const RayGenData& self = owl::getProgramData<RayGenData>();
    const vec2i pixelId = owl::getLaunchIndex();
    const int fbOfs = pixelId.x + self.frameBufferSize.x * pixelId.y;

    vec4f color = vec4f(optixLaunchParams.depthBuffer[fbOfs], 1);
//     bool used[NUM_BINS] = { false };
//     for (int i = -KERNEL_SIZE / 2; i < (KERNEL_SIZE / 2) + 1; i += 1) {
//         for (int j = -KERNEL_SIZE / 2; j < (KERNEL_SIZE / 2) + 1; j += 1) {
//             if (i == 0 && j == 0) {
//                 continue;
//             }
//             vec2i newPixel = pixelId + vec2i(i, j);
//             if ((newPixel.x < 0 || newPixel.x == self.frameBufferSize.x) ||
//                 (newPixel.y < 0 || newPixel.y == self.frameBufferSize.y)) {
//                     continue;
//                 }

//             int newFbOfs = newPixel.x + self.frameBufferSize.x * newPixel.y;
//             if (used[optixLaunchParams.binIdxBuffer[newFbOfs]]) {
//                 continue;
//             }

//             // Don't do spatial reuse if normals differ by more than 20 degrees
//             if (dot((vec3f)optixLaunchParams.normalBuffer[fbOfs], (vec3f)optixLaunchParams.normalBuffer[newFbOfs]) < cos(0.34)) {
//                 continue;
//             }

//             used[optixLaunchParams.binIdxBuffer[newFbOfs]] = true;
// #ifdef ACCUM
//             color = color + (vec4f)optixLaunchParams.accumBuffer[newFbOfs];
// #endif
//         }
//     }

    // self.frameBuffer[fbOfs] = owl::make_rgba(vec3f(
    //     linear_to_srgb(color.x),
    //     linear_to_srgb(color.y),
    //     linear_to_srgb(color.z)
    // ));
    self.frameBuffer[fbOfs] = make_rgba(color);
    // optixLaunchParams.accumBuffer[fbOfs] = color;
}
#endif