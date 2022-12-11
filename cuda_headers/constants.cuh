#pragma once

// Constants
#define PI 3.14159265359f
#define EPS 1e-6

// LTC parameters
#define MAX_LTC_LIGHTS 12
#define MAX_ELEMS 100

// Bloom Filter parameters
#define NUM_HASH 4   	// Number of Hash functions to use
#define NUM_LSB  8   	// Number of bits to consider from hash
#define NUM_BITS 8  	// Number of integers to use for bloom filter

// Renderer
#define DEBUG_DIFFUSE 0
#define DEBUG_ALPHA 1
#define DEBUG_SIL 2
#define LTC_BASE 3
#define LTC_SAMPLE_TRI 4
#define LTC_SAMPLE_POLY 5
#define DIRECT_LIGHTING 6

// #define RENDERER DEBUG_DIFFUSE
// #define RENDERER DEBUG_ALPHA
// #define RENDERER DEBUG_SIL
// #define RENDERER LTC_BASE
// #define RENDERER LTC_SAMPLE_TRI
#define RENDERER LTC_SAMPLE_POLY
// #define RENDERER DIRECT_LIGHTING

#define SAMPLES 1

// Features
// #define SIL                     // Whether to use integrate over silhouette or over all triangles of polygon
#define BSP_SIL                 // Whether to use BSP to calculate silhouette
// #define USE_BLOOM            // Whether to use bloom filters for set
// #define REJECTION_SAMPLING   // Whether to use rejection sampling or unique sampling