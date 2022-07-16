
static __device__
vec3f barycentricInterpolate(vec3f* tex, vec3i index)
{
    float u = optixGetTriangleBarycentrics().x;
    float v = optixGetTriangleBarycentrics().y;

    return (1.f - u - v) * tex[index.x]
        + u * tex[index.y]
        + v * tex[index.z];
}

static __device__
vec2f barycentricInterpolate(vec2f* tex, vec3i index)
{
    float u = optixGetTriangleBarycentrics().x;
    float v = optixGetTriangleBarycentrics().y;

    return (1.f - u - v) * tex[index.x]
        + u * tex[index.y]
        + v * tex[index.z];
}

static __device__
vec3f uniformSampleHemisphere(vec2f rand)
{
    float z = rand.x;
    float r = owl::sqrt(owl::max(0.f, 1.f - z * z));
    float phi = 2.f * PI * rand.y;

    return normalize(vec3f(r * cos(phi), r * sin(phi), z));
}

static __device__
vec3f apply_mat(vec3f mat[3], vec3f v)
{
    vec3f result(dot(mat[0], v), dot(mat[1], v), dot(mat[2], v));
    return result;
}

static __device__
void orthonormalBasis(vec3f n, vec3f mat[3], vec3f invmat[3])
{
    vec3f c1, c2, c3;
    if (n.z < -0.999999f)
    {
        c1 = vec3f(0, -1, 0);
        c2 = vec3f(-1, 0, 0);
    }
    else
    {
        float a = 1. / (1. + n.z);
        float b = -n.x * n.y * a;
        c1 = normalize(vec3f(1. - n.x * n.x * a, b, -n.x));
        c2 = normalize(vec3f(b, 1. - n.y * n.y * a, -n.y));
    }
    c3 = n;

    mat[0] = c1;
    mat[1] = c2;
    mat[2] = c3;

    invmat[0] = vec3f(c1.x, c2.x, c3.x);
    invmat[1] = vec3f(c1.y, c2.y, c3.y);
    invmat[2] = vec3f(c1.z, c2.z, c3.z);
}

static __device__
vec3f samplePointOnTriangle(vec3f v1, vec3f v2, vec3f v3,
    float u1, float u2)
{
    float su1 = owl::sqrt(u1);
    return (1 - su1) * v1 + su1 * ((1 - u2) * v2 + u2 * v3);
}