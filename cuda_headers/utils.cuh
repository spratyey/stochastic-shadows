#pragma once

__device__
vec3f barycentricInterpolate(vec3f* tex, vec3i index)
{
    float u = optixGetTriangleBarycentrics().x;
    float v = optixGetTriangleBarycentrics().y;

    return (1.f - u - v) * tex[index.x]
        + u * tex[index.y]
        + v * tex[index.z];
}

__device__
vec2f barycentricInterpolate(vec2f* tex, vec3i index)
{
    float u = optixGetTriangleBarycentrics().x;
    float v = optixGetTriangleBarycentrics().y;

    return (1.f - u - v) * tex[index.x]
        + u * tex[index.y]
        + v * tex[index.z];
}

__device__
vec3f uniformSampleHemisphere(vec2f rand)
{
    float z = rand.x;
    float r = owl::sqrt(owl::max(0.f, 1.f - z * z));
    float phi = 2.f * PI * rand.y;

    return normalize(vec3f(r * cos(phi), r * sin(phi), z));
}

__device__
vec2f ConcentricSampleDisk(vec2f rand) {
    // Map uniform random numbers to $[-1,1]^2$
    vec2f uOffset = 2.f * rand - vec2f(1, 1);

    // Handle degeneracy at the origin
    if (uOffset.x == 0 && uOffset.y == 0) return vec2f(0, 0);

    // Apply concentric mapping to point
    float theta, r;
    if (owl::abs(uOffset.x) > owl::abs(uOffset.y)) {
        r = uOffset.x;
        theta = PI / 4.f * (uOffset.y / uOffset.x);
    }
    else {
        r = uOffset.y;
        theta = PI / 2.f - PI / 4.f * (uOffset.x / uOffset.y);
    }
    return r * vec2f(owl::cos(theta), owl::sin(theta));
}

__device__
vec3f CosineSampleHemisphere(vec2f rand) {
    vec2f d = ConcentricSampleDisk(rand);
    float z = owl::sqrt(owl::max(0.f, 1.f - d.x * d.x - d.y * d.y));
    return normalize(vec3f(d.x, d.y, z));
}

__device__
vec3f apply_mat(vec3f mat[3], vec3f v)
{
    vec3f result(dot(mat[0], v), dot(mat[1], v), dot(mat[2], v));
    return result;
}

__device__
void matrixInverse(vec3f m[3], vec3f minv[3]) {
    int indxc[3], indxr[3];
    int ipiv[3] = { 0, 0, 0 };

    minv[0] = m[0];
    minv[1] = m[1];
    minv[2] = m[2];

    for (int i = 0; i < 3; i++) {
        int irow = 0, icol = 0;
        float big = 0.f;
        // Choose pivot
        for (int j = 0; j < 3; j++) {
            if (ipiv[j] != 1) {
                for (int k = 0; k < 3; k++) {
                    if (ipiv[k] == 0) {
                        if (abs(minv[j][k]) >= big) {
                            big = abs(minv[j][k]);
                            irow = j;
                            icol = k;
                        }
                    }
                }
            }
        }
        ++ipiv[icol];
        // Swap rows _irow_ and _icol_ for pivot
        if (irow != icol) {
            for (int k = 0; k < 3; ++k) {
                float temp = minv[irow][k];
                minv[irow][k] = minv[icol][k];
                minv[icol][k] = temp;
            }
        }
        indxr[i] = irow;
        indxc[i] = icol;

        // Set $m[icol][icol]$ to one by scaling row _icol_ appropriately
        float pivinv = 1.f / minv[icol][icol];
        minv[icol][icol] = 1.f;
        for (int j = 0; j < 3; j++) minv[icol][j] *= pivinv;

        // Subtract this row from others to zero out their columns
        for (int j = 0; j < 3; j++) {
            if (j != icol) {
                float save = minv[j][icol];
                minv[j][icol] = 0.f;
                for (int k = 0; k < 3; k++) minv[j][k] -= minv[icol][k] * save;
            }
        }
    }
    // Swap columns to reflect permutation
    for (int j = 2; j >= 0; j--) {
        if (indxr[j] != indxc[j]) {
            for (int k = 0; k < 3; k++) {
                float temp = minv[k][indxr[j]];
                minv[k][indxr[j]] = minv[k][indxc[j]];
                minv[k][indxc[j]] = temp;
            }
        }
    }
}

__device__
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

    matrixInverse(mat, invmat);
}

__device__
vec3f samplePointOnTriangle(vec3f v1, vec3f v2, vec3f v3,
    float u1, float u2)
{
    float su1 = owl::sqrt(u1);
    return (1 - su1) * v1 + su1 * ((1 - u2) * v2 + u2 * v3);
}

__device__
float sphericalTheta(vec3f p) {
    return acos(p.z);
}

__device__
float balanceHeuristic(int nf, float fPdf, int ng, float gPdf) {
    return (nf * fPdf) / (nf * fPdf + ng * gPdf);
}

__device__
float PowerHeuristic(int nf, float fPdf, int ng, float gPdf) {
    float f = nf * fPdf, g = ng * gPdf;
    return (f * f) / (f * f + g * g);
}