#pragma once

#include "owl/common/math/vec.h"
#include "common.cuh"
#include "frostbite.cuh"
#include "utils.cuh"

using namespace owl;

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
    vec3f wi = normalize(lpoint - si.p);
    vec3f wi_local = normalize(apply_mat(si.to_local, wi));

    float xmy = pow(owl::length(lpoint - si.p), 2.f);
    float lDotWi = owl::abs(owl::dot(lnormal, -wi));

    light_pdf = lightSelectionPdf * (xmy / (larea * lDotWi));

    ShadowRay ray;
    ray.origin = si.p + 1e-3f * si.n_geom;
    ray.direction = wi;

    ShadowRayData srd;
    owl::traceRay(optixLaunchParams.world, ray, srd);

    if (si.wo_local.z > 0.f && wi_local.z > 0.f && srd.visibility != vec3f(0.f) && light_pdf > 0.f && owl::dot(-wi, lnormal) > 0.f) {
        vec3f brdf = evaluate_brdf(si.wo_local, wi_local, si.diffuse, si.alpha);
        brdf_pdf = get_brdf_pdf(si.alpha, si.wo_local, normalize(si.wo_local + wi_local));

        if (mis && brdf_pdf > 0.f) {
            float weight = PowerHeuristic(1, light_pdf, 1, brdf_pdf);
            color += brdf * lemit * owl::abs(wi_local.z) * weight / light_pdf;
        }
        else if (!mis) {
            color += brdf * lemit * owl::abs(wi_local.z) / light_pdf;
        }
    }

    return color;
}


__device__
vec3f sampleBRDF(SurfaceInteraction si, float lightSelectionPdf, vec2f rand, bool mis)
{
    vec3f wi_local = sample_GGX(rand, si.alpha, si.wo_local);
    vec3f wi = normalize(apply_mat(si.to_world, wi_local));

    ShadowRay ray;
    ShadowRayData srd;
    ray.origin = si.p + 1e-3f * si.n_geom;
    ray.direction = wi;
    owl::traceRay(optixLaunchParams.world, ray, srd);

    vec3f color(0.f, 0.f, 0.f);
    float light_pdf = 0.f, brdf_pdf = 0.f;

    if (wi_local.z > 0.f && si.wo_local.z > 0.f && srd.visibility != vec3f(0.f)) {
        float xmy = pow(owl::length(srd.point - si.p), 2.f);
        float lDotWi = owl::abs(owl::dot(srd.normal, -wi));
        light_pdf = lightSelectionPdf * (xmy / (srd.area * lDotWi));

        vec3f brdf = evaluate_brdf(si.wo_local, wi_local, si.diffuse, si.alpha);
        brdf_pdf = get_brdf_pdf(si.alpha, si.wo_local, normalize(si.wo_local + wi_local));

        if (mis && light_pdf > 0.f && brdf_pdf > 0.f) {
            float weight = PowerHeuristic(1, brdf_pdf, 1, light_pdf);
            color += brdf * srd.emit * owl::abs(wi_local.z) * weight / brdf_pdf;
        }
        else if (!mis && brdf_pdf > 0.f) {
            color += brdf * srd.emit * owl::abs(wi_local.z) / brdf_pdf;
        }
    }

    return color;
}

__device__
vec3f estimateDirectLighting(SurfaceInteraction& si, LCGRand& rng, int type)
{
    vec3f color = vec3f(0.f);
    for (int i = 0; i < SAMPLES; i++) {
        vec3f tmpColor = vec3f(0.0);
        vec2f rand1 = vec2f(lcg_randomf(rng), lcg_randomf(rng));
        vec2f rand2 = vec2f(lcg_randomf(rng), lcg_randomf(rng));

        vec3f lightSample = vec3f(0.f);
        vec3f brdfSample = vec3f(0.f);

        if (type == 0) {
            int selectedTriLight = round(lcg_randomf(rng) * (optixLaunchParams.numTriLights - 1));
            float lightSelectionPdf = 1.f / optixLaunchParams.numTriLights;

            lightSample = sampleLightSource(si, selectedTriLight, lightSelectionPdf, rand1, false);

            tmpColor = lightSample;
        }
        else if (type == 1) {
            brdfSample = sampleBRDF(si, 0.f, rand2, false);

            tmpColor = brdfSample;
        }
        else if (type == 2) {
            int selectedTriLight = round(lcg_randomf(rng) * (optixLaunchParams.numTriLights - 1));
            float lightSelectionPdf = 1.f / optixLaunchParams.numTriLights;

            brdfSample = sampleBRDF(si, lightSelectionPdf, rand1, true);
            lightSample = sampleLightSource(si, selectedTriLight, lightSelectionPdf, rand2, true);

            tmpColor = brdfSample + lightSample;
        }

        // Make sure there are no negative colors!
        color.x += owl::max(0.f, tmpColor.x);
        color.y += owl::max(0.f, tmpColor.y);
        color.z += owl::max(0.f, tmpColor.z);
    }

    return color / SAMPLES;
}