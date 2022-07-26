#pragma once 

__device__
float G2(float alpha, vec3f V, vec3f L) {
    float alphasq = alpha * alpha;

    float cosV = V.z;
    float cosV2 = cosV * cosV;

    float cosL = L.z;
    float cosL2 = cosL * cosL;

    float num = 2.f * cosV * cosL;
    float denom = cosV * owl::sqrt(alphasq + (1.f - alphasq) * cosL2) + cosL * owl::sqrt(alphasq + (1.f - alphasq) * cosV2);

    return num / denom;
}

__device__
float D(float alpha, vec3f H) {
    float alphasq = alpha * alpha;

    float num = alphasq;
    float denom = PI * pow(H.z * H.z * (alphasq - 1.f) + 1, 2.f);

    return num / denom;
}

__device__
float GGX(float alpha, vec3f V, vec3f L) {
    vec3f H = normalize(V + L);
    float value = D(alpha, H) * G2(alpha, V, L) / 4.0f / V.z / L.z;

    return value;
}

/*! Evaluates the full BRDF with both diffuse and specular terms.
    The specular BRDF is GGX specular (Taken from Eric Heitz's JCGT paper).
    Fresnel is not used (commented).
    Evaluates only f (i.e. BRDF without cosine foreshortening) */
__device__
vec3f evaluate_brdf(vec3f wo, vec3f wi, vec3f diffuse_color, float alpha) {
    vec3f brdf = vec3f(0.0f);

    // Diffuse + specular
    brdf += diffuse_color / PI;
    brdf += GGX(alpha, wo, wi);

    return brdf;
}

__device__
float get_brdf_pdf(float alpha, vec3f V, vec3f Ne) {
    float cosT = Ne.z;
    float alphasq = alpha * alpha;

    float num = alphasq * cosT;
    float denom = PI * pow((alphasq - 1.f) * cosT * cosT + 1.f, 2.f);

    float pdf = num / denom;
    return pdf / (4.f * dot(V, Ne));
}

__device__
vec3f sample_GGX(vec2f rand, float alpha, vec3f V) {
    float num = 1.f - rand.x;
    float denom = rand.x * (alpha * alpha - 1.f) + 1;
    float t = acos(owl::sqrt(num / denom));
    float p = 2.f * PI * rand.y;

    vec3f N(sin(t) * cos(p), sin(t) * sin(p), cos(t));
    vec3f L = -V + 2.0f * N * dot(V, N);

    return normalize(L);
}