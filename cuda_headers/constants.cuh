#pragma once

// Constants
#define PI 3.1415926f
#define EPS 1e-6

// LTC parameters
#define MAX_LTC_LIGHTS 10

// Bloom Filter parameters
#define NUM_HASH 4   	// Number of Hash functions to use
#define NUM_LSB  8   	// Number of bits to consider from hash
#define NUM_BITS 8  	// Number of integers to use for bloom filter

// #define DEBUG

// Renderer
#define DEBUG_DIFFUSE 0
#define DEBUG_SIL 1
#define LTC_BASE 2
#define LTC_SAMPLE_TRI 3
#define LTC_SAMPLE_POLY 4
#define DIRECT_LIGHTING 5

// #define RENDERER DEBUG_DIFFUSE
// #define RENDERER DEBUG_SIL
// #define RENDERER LTC_BASE
// #define RENDERER LTC_SAMPLE_TRI
#define RENDERER LTC_SAMPLE_POLY
// #define RENDERER DIRECT_LIGHTING

// Features
#define BSP_SIL
#define USE_BLOOM
// #define REJECTION_SAMPLING
#define SAMPLES 64