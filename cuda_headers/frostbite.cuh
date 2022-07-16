/*
Roughly frostbite BRDF, without fresnes and disney diffuse terms 
Link: https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
*/

static __device__
vec3f fresnel_schlick(vec3f fresnel_0, vec3f fresnel_90, float cos_theta) {
    float flipped = 1.0f - cos_theta;
    float flipped_squared = flipped * flipped;
    return fresnel_0 + (fresnel_90 - fresnel_0) * (flipped_squared * flipped * flipped_squared);
}

static __device__
float D(float alphax, float alphay, vec3f N) {
    float t1 = N.x / alphax;
    float t2 = N.y / alphay;
    float t3 = N.z;

    float value = PI * alphax * alphay * pow(t1 * t1 + t2 * t2 + t3 * t3, 2.0f);
    return 1.0f / value;
}

static __device__
float Lambda(float alphax, float alphay, vec3f V) {
    float t1 = V.x * alphax;
    float t2 = V.y * alphay;
    float t3 = V.z;
    float t4 = owl::sqrt(1.0f + (t1 * t1 + t2 * t2) / (t3 * t3));

    return 0.5f * (-1.0f + t4);
}

static __device__
float G1(float alphax, float alphay, vec3f V) {
    if (V.z <= 0.0f)
        return 0.0f;

    float value = 1.0f / (1.0f + Lambda(alphax, alphay, V));
    return value;
}

static __device__
float G2(float alphax, float alphay, vec3f V, vec3f L) {
    if (V.z <= 0.0f || L.z <= 0.0f)
        return 0.0f;

    float value = 1.0f / (1.0f + Lambda(alphax, alphay, V) + Lambda(alphax, alphay, L));
    return value;
}

static __device__
float Dv(float alphax, float alphay, vec3f V, vec3f Ne)
{
    float g1 = G1(alphax, alphay, V);
    float m = max(0.f, dot(V, Ne));
    float d = D(alphax, alphay, Ne);

    return g1 * m * d / V.z;
}

static __device__
float GGX(float alphax, float alphay, vec3f V, vec3f L) {
    vec3f H = normalize(V + L);
    float value = D(alphax, alphay, H) * G2(alphax, alphay, V, L) / 4.0f / V.z / L.z;

    return value;
}

/*! Evaluates the full BRDF with both diffuse and specular terms.
    The specular BRDF is GGX specular (Taken from Eric Heitz's JCGT paper).
    Fresnel is not used (commented).
    Evaluates only f (i.e. BRDF without cosine foreshortening) */
static __device__
vec3f evaluate_brdf(vec3f wo, vec3f wi, vec3f diffuse_color, float alpha) {
    // A few computations are shared between diffuse and specular evaluation
    vec3f half_vector = normalize(wo + wi);
    float lambert_incoming = wo.z;
    float lambert_outgoing = wi.z;
    float outgoing_dot_half = dot(wi, half_vector);
    vec3f brdf = vec3f(0.0f);

    // Diffuse + specular
    brdf += diffuse_color / PI;
    brdf += GGX(alpha, alpha, wo, wi);

    return brdf;
}

static __device__
float get_brdf_pdf(float alphax, float alphay, vec3f V, vec3f Ne) {
    return Dv(alphax, alphay, V, Ne) / (4.f * dot(V, Ne));
}

static __device__
vec3f sample_VNDF(float alphax, float alphay, vec3f V, vec2f rand_num) {
    float U1 = rand_num.x;
    float U2 = rand_num.y;

    vec3f Vh = normalize(V * vec3f(alphax, alphay, 1.0f));

    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    vec3f T1;
    if (lensq > 0.0f)
        T1 = vec3f(-Vh.y, Vh.x, 0.0) / owl::sqrt(lensq);
    else
        T1 = vec3f(1.0f, 0.0f, 0.0f);

    vec3f T2 = cross(Vh, T1);

    float r = owl::sqrt(U1);
    float phi = 2.0f * PI * U2;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * owl::sqrt(1.0f - t1 * t1) + s * t2;

    vec3f Nh = t1 * T1 + t2 * T2 + owl::sqrt(max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;

    vec3f Ne = vec3f(alphax * Nh.x, alphay * Nh.y, max(0.0f, Nh.z));
    Ne = normalize(Ne);

    return Ne;
}

static __device__
vec3f sample_GGX(float rand, float alphax, float alphay, vec3f V, float& pdf) {
    vec3f N = sample_VNDF(alphax, alphay, V, rand);
    vec3f L = -V + 2.0f * N * dot(V, N);

    pdf = get_brdf_pdf(alphax, alphay, V, N);

    return normalize(L);
}