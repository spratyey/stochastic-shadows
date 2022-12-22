#pragma once

// Constants
#define PI 3.14159265359f
#define EPS 1e-6

// Spatial reuse parameters
#define NUM_BINS 9          // Number of bins in which the polygons of each light should be divided
#define NUM_PASSES 2        // Number of spatial reuse passes
#define SPATIAL_SAMPLES 5   // Number of spatial neighbours to sample
#define KERNEL_SIZE 30      // NxN sized kernel

// LTC parameters
#define MAX_LTC_LIGHTS 25
#define MAX_ELEMS 400

// RIS parameters
#define NUM_PROPOSALS 32    // Number of proposals (M) to use for RIS 

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
#define LTC_MONTE_CARLO 6
#define DIRECT_LIGHTING 7
#define DIRECT_LIGHTING_RESTIR 8

// #define RENDERER DEBUG_DIFFUSE
// #define RENDERER DEBUG_ALPHA
// #define RENDERER DEBUG_SIL
// #define RENDERER LTC_BASE
// #define RENDERER LTC_SAMPLE_TRI
// #define RENDERER LTC_SAMPLE_POLY
// #define RENDERER LTC_MONTE_CARLO
// #define RENDERER DIRECT_LIGHTING
#define RENDERER DIRECT_LIGHTING_RESTIR

#define SAMPLES 1   // Number of samples per frame

#if RENDERER == DIRECT_LIGHTING_RESTIR
#define USE_RESERVOIRS      // Whether to use WRS for sampling
#endif

// Whether to enable accumulation buffer
#if RENDERER == DIRECT_LIGHTING || RENDERER == DIRECT_LIGHTING_RESTIR || RENDERER == LTC_MONTE_CARLO
#define ACCUM
#endif

// Features
#if RENDERER == LTC_SAMPLE_POLY || RENDERER == LTC_BASE || (RENDERER == DIRECT_LIGHTING_RESTIR && defined(USE_RESERVOIRS))
#define SPATIAL_REUSE
#endif

// #define SIL                     // Whether to use integrate over silhouette or over all triangles of polygon
// #define BSP_SIL                 // Whether to use BSP to calculate silhouette
// #define USE_BLOOM            // Whether to use bloom filters for set
// #define REJECTION_SAMPLING   // Whether to use rejection sampling or unique sampling