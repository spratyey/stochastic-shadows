#include "common.cuh"
#include "utils.cuh"

// These headers depend on functions included in common.cuh and utils.cuh
#include "ltc_many_lights_cuda.cuh"
#include "frostbite.cuh"

#include "ltc_utils.cuh"
#include "polygon_utils.cuh"

inline __device__ vec3f estimateDirectLighting(SurfaceInteraction& si);
inline __device__ vec3f estimateDirectLightingLBVH(SurfaceInteraction& si);
inline __device__ vec3f ltcDirecLighingBaseline(SurfaceInteraction& si);

inline __device__ vec3f multipleImportanceSampling(SurfaceInteraction& si, int selectedAreaLight, float lightSelectionPdf, vec2f rand1, vec2f rand2);

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
    }
    else {
        srd.visibility = vec3f(0.f);
    }
}

OPTIX_CLOSEST_HIT_PROGRAM(triangleMeshCH)()
{
    const TriangleMeshData& self = owl::getProgramData<TriangleMeshData>();
    const vec3i primitiveIndices = self.index[optixGetPrimitiveIndex()];
    vec3f& color = owl::getPRD<vec3f>();

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

    if (optixLaunchParams.rendererType == DIFFUSE)
        color = si.diffuse;
    else if (optixLaunchParams.rendererType == ALPHA)
        color = si.alpha;
    else if (optixLaunchParams.rendererType == NORMALS)
        color = 0.5f * (si.n_geom + 1.f);
    else if (optixLaunchParams.rendererType == DIRECT_LIGHT) {
        if (self.isLight)
            color = self.emit;
        else
            color = estimateDirectLighting(si);
    }
    else if (optixLaunchParams.rendererType == DIRECT_LIGHT_LBVH) {
        if (self.isLight)
            color = self.emit;
        else
            color = estimateDirectLightingLBVH(si);
    }
    else if (optixLaunchParams.rendererType == LTC_BASELINE) {
        if (self.isLight)
            color = self.emit;
        else
            color = ltcDirecLighingBaseline(si);
    }
}

OPTIX_MISS_PROGRAM(miss)()
{
    const vec2i pixelId = owl::getLaunchIndex();

    const MissProgData& self = owl::getProgramData<MissProgData>();

    vec3f& prd = owl::getPRD<vec3f>();
    prd = self.const_color;
}

inline __device__
vec3f ltcDirecLighingBaseline(SurfaceInteraction& si)
{
    const vec2i pixelId = owl::getLaunchIndex();
    owl::common::LCG<4> rng(pixelId.x * pixelId.y, optixLaunchParams.accumId);

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

    for (int lidx = 0; lidx < optixLaunchParams.numAreaLights; lidx++) {
        vec3f lv1 = optixLaunchParams.areaLights[lidx].v1;
        vec3f lv2 = optixLaunchParams.areaLights[lidx].v2;
        vec3f lv3 = optixLaunchParams.areaLights[lidx].v3;
        vec3f lemit = optixLaunchParams.areaLights[lidx].emissionRadiance;
        vec3f lnormal = optixLaunchParams.areaLights[lidx].normal;
        float larea = optixLaunchParams.areaLights[lidx].area;

        // Move to origin and normalize
        lv1 = owl::normalize(lv1 - si.p);
        lv2 = owl::normalize(lv2 - si.p);
        lv3 = owl::normalize(lv3 - si.p);

        vec3f cg = normalize(lv1 + lv2 + lv3);
        if (owl::dot(-cg, lnormal) < 0.f)
            continue;
        
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

        diff_clipped[0] = owl::normalize(apply_mat(ltc_mat_inv, diff_clipped[0]));
        diff_clipped[1] = owl::normalize(apply_mat(ltc_mat_inv, diff_clipped[1]));
        diff_clipped[2] = owl::normalize(apply_mat(ltc_mat_inv, diff_clipped[2]));
        diff_clipped[3] = owl::normalize(apply_mat(ltc_mat_inv, diff_clipped[3]));
        diff_clipped[4] = owl::normalize(apply_mat(ltc_mat_inv, diff_clipped[4]));

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

        color += (si.diffuse * lemit * diffuse_shading) + (amplitude * lemit * ggx_shading);
    }

    return color;
}

inline __device__
vec3f estimateDirectLightingLBVH(SurfaceInteraction& si)
{
    const vec2i pixelId = owl::getLaunchIndex();
    owl::common::LCG<4> rng(pixelId.x * pixelId.y, optixLaunchParams.accumId);

    int nodeIdx = 0;
    float lightSelectionPdf = 1.f;
    int selectedAreaLight = 0;
    for (int i = 0; i < optixLaunchParams.areaLightsBVHHeight + 1; i++) {
        LightBVH node = optixLaunchParams.areaLightsBVH[nodeIdx];
        
        // If leaf
        if (node.left == 0 && node.right == 0) {
            selectedAreaLight = node.primIdx + round(rng() * node.primCount);
            lightSelectionPdf *= 1.f / node.primCount;

            break;
        }

        LightBVH leftNode = optixLaunchParams.areaLightsBVH[node.left];
        LightBVH rightNode = optixLaunchParams.areaLightsBVH[node.right];

        float leftImp = 1.f / pow(owl::length(leftNode.aabbMid - si.p), 2.f);
        float rightImp = 1.f / pow(owl::length(rightNode.aabbMid - si.p), 2.f);
        float sum = leftImp + rightImp;

        leftImp = leftImp / sum;
        rightImp = rightImp / sum;

        float eps = rng();
        if (eps < leftImp) {
            nodeIdx = node.left;
            lightSelectionPdf *= leftImp;
        }
        else {
            nodeIdx = node.right;
            lightSelectionPdf *= rightImp;
        }
    }

    vec2f rand1 = vec2f(rng(), rng());
    vec2f rand2 = vec2f(rng(), rng());

    vec3f color = multipleImportanceSampling(si, selectedAreaLight, lightSelectionPdf, rand1, rand2);

    color.x = owl::max(0.f, color.x);
    color.y = owl::max(0.f, color.y);
    color.z = owl::max(0.f, color.z);

    return color;
}

inline __device__
vec3f estimateDirectLighting(SurfaceInteraction& si)
{
    const vec2i pixelId = owl::getLaunchIndex();
    owl::common::LCG<16> rng(pixelId.x * pixelId.y, optixLaunchParams.accumId);

    int selectedAreaLight = round(rng() * optixLaunchParams.numAreaLights);
    float lightSelectionPdf = 1.f / optixLaunchParams.numAreaLights;

    vec2f rand1 = vec2f(rng(), rng());
    vec2f rand2 = vec2f(rng(), rng());

    vec3f color = multipleImportanceSampling(si, selectedAreaLight, lightSelectionPdf, rand1, rand2);

    color.x = owl::max(0.f, color.x);
    color.y = owl::max(0.f, color.y);
    color.z = owl::max(0.f, color.z);

    return color;
}

inline __device__
vec3f multipleImportanceSampling(SurfaceInteraction& si, int selectedAreaLight, float lightSelectionPdf, vec2f rand1, vec2f rand2)
{
    TriLight areaLight = optixLaunchParams.areaLights[selectedAreaLight];

    vec3f color(0.f, 0.f, 0.f);
    vec3f wo_local = normalize(apply_mat(si.to_local, si.wo));
    vec3f normal_local(0.f, 0.f, 1.f);

    {
        float light_pdf, brdf_pdf;

        vec3f lv1 = areaLight.v1;
        vec3f lv2 = areaLight.v2;
        vec3f lv3 = areaLight.v3;
        vec3f lnormal = areaLight.normal;
        vec3f lemit = areaLight.emissionRadiance;
        float larea = areaLight.area;

        vec3f lpoint = samplePointOnTriangle(lv1, lv2, lv3, rand1.x, rand1.y);
        vec3f wi = normalize(lpoint - si.p);
        vec3f wi_local = normalize(apply_mat(si.to_local, wi));

        if (owl::dot(-wi, lnormal) < 0.f)
            return color;

        float xmy = pow(owl::length(lpoint - si.p), 2.f);
        float lDotWi = owl::abs(owl::dot(lnormal, -wi));

        light_pdf = lightSelectionPdf * (xmy / (larea * lDotWi));

        if (light_pdf <= 0.f)
            return color;

        ShadowRay ray;
        ray.origin = si.p + 1e-3f * si.n_geom;
        ray.direction = wi;

        ShadowRayData srd;
        owl::traceRay(optixLaunchParams.world, ray, srd);

        if (wo_local.z > 0.f && wi_local.z > 0.f && srd.visibility != vec3f(0.f)) {
            vec3f brdf = evaluate_brdf(wo_local, wi_local, si.diffuse, si.alpha);
            brdf_pdf = get_brdf_pdf(si.alpha, si.alpha, wo_local, normalize(wo_local + wi_local));
            float weight = balanceHeuristic(1, light_pdf, 1, brdf_pdf);

            color += brdf * lemit * owl::abs(wi_local.z) * weight / light_pdf;
        }
    }

    {
        float light_pdf, brdf_pdf;

        vec3f wi_local = sample_GGX(rand2, si.alpha, si.alpha, wo_local, brdf_pdf);
        vec3f wi = normalize(apply_mat(si.to_world, wi_local));

        if (brdf_pdf <= 0.f)
            return color;

        ShadowRay ray;
        ray.origin = si.p + 1e-3f * si.n_geom;
        ray.direction = wi;

        ShadowRayData srd;
        owl::traceRay(optixLaunchParams.world, ray, srd);

        if (wi_local.z > 0.f && wo_local.z > 0.f && srd.visibility != vec3f(0.f)) {
            float xmy = pow(owl::length(srd.point - si.p), 2.f);
            float lDotWi = owl::abs(owl::dot(srd.normal, -wi));
            light_pdf = lightSelectionPdf * (xmy / (srd.area * lDotWi));

            if (light_pdf <= 0.f)
                return color;

            vec3f brdf = evaluate_brdf(wo_local, wi_local, si.diffuse, si.alpha);
            float weight = balanceHeuristic(1, brdf_pdf, 1, light_pdf);

            color += brdf * srd.emit * owl::abs(wi_local.z) * weight / brdf_pdf;
        }

    }

    return color;
}

