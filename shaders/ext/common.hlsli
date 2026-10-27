// common.hlsli -- shared constants and helpers for D3D11 PyroWave shaders

#ifndef COMMON_HLSLI
#define COMMON_HLSLI

// Feature level 11_0, cs_5_0: no 16-bit types, no wave intrinsics.
// All groupshared staging arrays must have a barrier before reuse.

#define MAX_GROUPS 1024

#endif /* COMMON_HLSLI */
