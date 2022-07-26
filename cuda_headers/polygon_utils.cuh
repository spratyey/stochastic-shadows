#pragma once

/*! Returns the intersection of the line connecting the given two points with
	the plane z == 0.0f.*/
__device__
vec3f iz0(vec3f lhs, vec3f rhs)
{
	float lerp_factor = lhs.z / (lhs.z - rhs.z);
	// Equivalent to the following but I have trust issues regarding the
	// stability of mix()
	// return vec3f(mix(lhs.xy, rhs.xy, lerp_factor), 0.0f);

	// return vec3f(fma(vec2(lerp_factor), rhs.xy, fma(-vec2(lerp_factor), lhs.xy, lhs.xy)), 0.0f);

	vec2f a(lerp_factor, lerp_factor);
	vec2f b(rhs.x, rhs.y);
	vec2f c = -vec2f(lerp_factor, lerp_factor) * vec2f(lhs.x, lhs.y) + vec2f(lhs.x, lhs.y);
	vec2f temp = a * b + c;

	return normalize(vec3f(temp.x, temp.y, 0.f));
}


/*! This function clips the given convex polygon with vertices in v to the
	upper hemisphere (i.e. the half-space with non-negative z-coordinate).
	The vertex count after clipping is returned. It is either zero or between
	three and vertex_count + 1. If it is less than MAX_POLYGON_VERTEX_COUNT,
	the first entry of v is repeated at the returned vertex count for the
	output. vertex_count must be at least
	MIN_POLYGON_VERTEX_COUNT_BEFORE_CLIPPING.*/
__device__
int clipPolygon(int vertex_count, vec3f v[5])
{
	if (vertex_count == 0)
		return 0;
	// The vertex count after clipping
	int vc;
	// Encode the whole configuration into a single integer
	int bit_mask = vertex_count;
	[[unroll]]
	for (int i = 0; i != 5 - 1; ++i)
		bit_mask |= (v[i].z > 0.0f && (i < 3 || i < vertex_count)) ? (1 << (i + 3)) : 0;
	// This code has been generated automatically to handle all possible cases
	// with a single conditional jump and no unnecessary instructions
	switch (bit_mask) {
		// case    3:   vc = 0;   break;
		// case   59:   vc = 3;   v[3] = v[0];   break;
		// case   11:   vc = 3;   v[1] = iz0(v[0], v[1]);   v[2] = iz0(v[2], v[0]);   v[3] = v[0];   break;
		// case   19:   vc = 3;   v[0] = iz0(v[0], v[1]);   v[2] = iz0(v[1], v[2]);   v[3] = v[0];   break;
		// case   35:   vc = 3;   v[0] = iz0(v[2], v[0]);   v[1] = iz0(v[1], v[2]);   v[3] = v[0];   break;
		// case   27:   vc = 4;   v[3] = iz0(v[2], v[0]);   v[2] = iz0(v[1], v[2]);   v[4] = v[0];   break;
		// case   51:   vc = 4;   v[3] = iz0(v[2], v[0]);   v[0] = iz0(v[0], v[1]);   v[4] = v[0];   break;
		// case   43:   vc = 4;   v[3] = v[2];   v[2] = iz0(v[1], v[2]);   v[1] = iz0(v[0], v[1]);   v[4] = v[0];   break;
		// case    4:   vc = 0;   break;
		// case  124:   vc = 4;   v[4] = v[0];   break;
		// case   12:   vc = 3;   v[1] = iz0(v[0], v[1]);   v[2] = iz0(v[3], v[0]);   v[3] = v[0];   break;
		// case   20:   vc = 3;   v[0] = iz0(v[0], v[1]);   v[2] = iz0(v[1], v[2]);   v[3] = v[0];   break;
		// case   36:   vc = 3;   v[0] = iz0(v[2], v[3]);   v[1] = iz0(v[1], v[2]);   v[3] = v[0];   break;
		// case   68:   vc = 3;   v[1] = iz0(v[3], v[0]);   v[0] = v[3];   v[2] = iz0(v[2], v[3]);   break;
		// case   28:   vc = 4;   v[2] = iz0(v[1], v[2]);   v[3] = iz0(v[3], v[0]);   v[4] = v[0];   break;
		// case   52:   vc = 4;   v[0] = iz0(v[0], v[1]);   v[3] = iz0(v[2], v[3]);   v[4] = v[0];   break;
		// case  100:   vc = 4;   v[0] = iz0(v[3], v[0]);   v[1] = iz0(v[1], v[2]);   v[4] = v[0];   break;
		// case   76:   vc = 4;   v[1] = iz0(v[0], v[1]);   v[2] = iz0(v[2], v[3]);   v[4] = v[0];   break;
		// case   60:   vc = 5;   v[4] = iz0(v[3], v[0]);   v[3] = iz0(v[2], v[3]);   break;
		// case  116:   vc = 5;   v[4] = iz0(v[3], v[0]);   v[0] = iz0(v[0], v[1]);   break;
		// case  108:   vc = 5;   v[4] = v[0];   v[0] = iz0(v[0], v[1]);   v[1] = iz0(v[1], v[2]);   break;
		// case   92:   vc = 5;   v[4] = v[3];   v[3] = iz0(v[2], v[3]);   v[2] = iz0(v[1], v[2]);   break;

	case    3:   vc = 0;   break;
	case   59:   vc = 3;   v[3] = v[0];   break;
	case   11:   vc = 3;   v[1] = iz0(v[0], v[1]);   v[2] = iz0(v[2], v[0]);   v[3] = v[0];   break;
	case   19:   vc = 3;   v[0] = iz0(v[0], v[1]);   v[2] = iz0(v[1], v[2]);   v[3] = v[0];   break;
	case   35:   vc = 3;   v[0] = iz0(v[2], v[0]);   v[1] = iz0(v[1], v[2]);   v[3] = v[0];   break;
	case   27:   vc = 4;   v[3] = iz0(v[2], v[0]);   v[2] = iz0(v[1], v[2]);   v[4] = v[0];   break;
	case   51:   vc = 4;   v[3] = iz0(v[2], v[0]);   v[0] = iz0(v[0], v[1]);   v[4] = v[0];   break;
	case   43:   vc = 4;   v[3] = v[2];   v[2] = iz0(v[1], v[2]);   v[1] = iz0(v[0], v[1]);   v[4] = v[0];   break;
	case    4:   vc = 0;   break;
	case  124:   vc = 4;   v[4] = v[0];   break;
	case   12:   vc = 3;   v[1] = iz0(v[0], v[1]);   v[2] = iz0(v[3], v[0]);   v[3] = v[0];   break;
	case   20:   vc = 3;   v[0] = iz0(v[0], v[1]);   v[2] = iz0(v[1], v[2]);   v[3] = v[0];   break;
	case   36:   vc = 3;   v[0] = iz0(v[2], v[3]);   v[1] = iz0(v[1], v[2]);   v[3] = v[0];   break;
	case   68:   vc = 3;   v[1] = iz0(v[3], v[0]);   v[0] = v[3];   v[2] = iz0(v[2], v[3]);   break;
	case   28:   vc = 4;   v[2] = iz0(v[1], v[2]);   v[3] = iz0(v[3], v[0]);   v[4] = v[0];   break;
	case   52:   vc = 4;   v[0] = iz0(v[0], v[1]);   v[3] = iz0(v[2], v[3]);   v[4] = v[0];   break;
	case  100:   vc = 4;   v[0] = iz0(v[3], v[0]);   v[1] = iz0(v[1], v[2]);   v[4] = v[0];   break;
	case   76:   vc = 4;   v[1] = iz0(v[0], v[1]);   v[2] = iz0(v[2], v[3]);   v[4] = v[0];   break;
	case   60:   vc = 5;   v[4] = iz0(v[3], v[0]);   v[3] = iz0(v[2], v[3]);   break;
	case  116:   vc = 5;   v[4] = iz0(v[3], v[0]);   v[0] = iz0(v[0], v[1]);   break;
	case  108:   vc = 5;   v[4] = v[0];   v[0] = iz0(v[0], v[1]);   v[1] = iz0(v[1], v[2]);   break;
	case   92:   vc = 5;   v[4] = v[3];   v[3] = iz0(v[2], v[3]);   v[2] = iz0(v[1], v[2]);   break;
	default:
		// This should never happen. Just pretend the polygon is below the
		// horizon.
		vc = 0;
		break;
	};
	return vc;
}