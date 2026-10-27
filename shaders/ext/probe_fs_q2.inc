// T-Q-5 probe results: FS-Q2 division values
// Computed on HYTE (MSVC) and verified on 192.168.0.24 (gcc 15.3.0) - bit-exact
// FS-Q1 boundary probe: see probe_cpu.cpp for test generation logic
// OQ-3 f16 rounding: CPU RNE reference below; GPU test infrastructure in tools/f16_rounding_test.cpp

// FS-Q2: 16 qs values = 1/(code/8 + 0.25), CPU correctly-rounded reference
static const float fs_q2_qs_expected[16] = {
    4.0f,           // code=0: 1/0.25 = 4.0 (exact)
    2.6666667461f,  // code=1: 1/0.375 = 8/3 (not exact in f32)
    2.0f,           // code=2: 1/0.5 = 2.0 (exact)
    1.6000000238f,  // code=3: 1/0.625 = 8/5 (not exact)
    1.3333333731f,  // code=4: 1/0.75 = 4/3 (not exact)
    1.1428571939f,  // code=5: 1/0.875 = 8/7 (not exact)
    1.0f,           // code=6: 1/1.0 = 1.0 (exact)
    0.8888888955f,  // code=7: 1/1.125 = 8/9 (not exact)
    0.8000000119f,  // code=8: 1/1.25 = 4/5 (not exact)
    0.7272727489f,  // code=9: 1/1.375 = 8/11 (not exact)
    0.6666666865f,  // code=10: 1/1.5 = 2/3 (not exact)
    0.6153846383f,  // code=11: 1/1.625 = 8/13 (not exact)
    0.5714285970f,  // code=12: 1/1.75 = 4/7 (not exact)
    0.5333333611f,  // code=13: 1/1.875 = 8/15 (not exact)
    0.5f,           // code=14: 1/2.0 = 0.5 (exact)
    0.4705882370f,  // code=15: 1/2.125 = 8/17 (not exact)
};

// Hex float form for HLSL embedding
// Note: in HLSL context this produces 'uint' array; in C++ it produces 'uint32_t' array
static const uint fs_q2_qs_bits[16] = {
    0x40800000u, 0x402aaaabu, 0x40000000u, 0x3fcccccdu,
    0x3faaaaabu, 0x3f924925u, 0x3f800000u, 0x3f638e39u,
    0x3f4ccccdu, 0x3f3a2e8cu, 0x3f2aaaabu, 0x3f1d89d9u,
    0x3f124925u, 0x3f088889u, 0x3f000000u, 0x3ef0f0f1u
};

// T-Q-5 conclusion:
// All 16 qs values are correctly rounded by MSVC (and gcc).
// The GPU (RTX Vulkan / D3D11) should produce identical values since
// 1.0f / x for these 16 inputs is a single division with IEEE rounding.
// FS-Q1 (m_wave / target) is the more concerning point - see FS-Q1 probe.
