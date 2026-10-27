// quant.hlsl -- Quantization stage: per-block quant scale, quantize, stats, payload
// Port of R/shaders/wavelet_quant.comp to cs_5_0
//
// Spec: docs/stages/02_quant.md
// Source: R/shaders/wavelet_quant.comp (wq:N), R/shaders/dwt_quant_scale.h,
//         R/shaders/constants.h, R/pyrowave_encoder.cpp:28-38,111-176,419-492
//
// Structure: five barriers, no barrier inside the q loop (§5.3)

#include "common.hlsli"
#include "../probe_fs_q2.inc"   // fs_q2_qs_bits[16]

// ------------------------------------------------------------------
// Push constants (matches QuantizerPushData in R/pyrowave_encoder.cpp:28-38)
// ------------------------------------------------------------------
// One dispatch covers every band of one coefficient buffer (at most four).
// w[0] = resolution.xy, 8x8 block counts
// w[1] = inv_resolution bits, coeff pitch, band count
// w[2 + 2*band] = layer, quant_resolution bits, block_offset, block_stride
// w[3 + 2*band] = rdo bits, pad
cbuffer PushConstants : register(b0) {
    uint4 w[10];
};

static int2 resolution;
static int2 resolution_8x8_blocks;
static uint coeff_pitch;
static float quant_resolution;
static int block_offset;
static int block_stride;
static float rdo_distortion_scale;
static int input_layer;

// ------------------------------------------------------------------
// Resources
// ------------------------------------------------------------------
ByteAddressBuffer t0 : register(t0);
RWByteAddressBuffer u0 : register(u0);
RWByteAddressBuffer u1 : register(u1);
RWByteAddressBuffer u2 : register(u2);

// ------------------------------------------------------------------
// Constants from constants.h
// ------------------------------------------------------------------
#define Q_PLANES_OFFSET 16
#define Q_PLANES_BITS   4
#define QUANT_SCALE_OFFSET 20
#define QUANT_SCALE_BITS   4

// ------------------------------------------------------------------
// Groupshared memory (§5.3)
// ------------------------------------------------------------------
groupshared float  gs_max[128];
groupshared float  gs_tex[128][8];
groupshared int    gs_b4x2[128];
groupshared int    gs_early[128];
groupshared int    gs_ctotal[16];
groupshared int    gs_group_base[1];
// 128 words per 8x8 cluster. Sixteen clusters share the group.
groupshared uint   gs_staging[2048];
groupshared int    gs_overflow[1];
groupshared uint   gs_payload_counter[1]; // groupshared counter for payload bytes
groupshared uint   gs_pcode[128];

// firstbithigh(0) is -1. On the HD 6670 it is not, so a 4x2 of zeros inside a
// live 8x8 (the partial block at the bottom of a 512x288 LL) was charged an
// extra magnitude bit. The decoder then scrambled that row. A full block never
// asks for the MSB of zero, which is why 720p and a 512x256 frame stayed a ramp.
int msb_of(uint v)
{
    if (v == 0u) return -1;
    return int(firstbithigh(v));
}

// Coefficient buffer written by the DWT store. Out-of-range texels are 0,
// matching Texture2D.Load.
float band_at(int layer, int x, int y)
{
    if ((uint)x >= (uint)resolution.x || (uint)y >= (uint)resolution.y) return 0.0f;
    uint pitch_h = (uint(resolution.y) + 31u) & ~31u;
    uint idx = uint(layer) * coeff_pitch * pitch_h + uint(y) * coeff_pitch + uint(x);
    return asfloat(t0.Load(idx * 4u));
}

[numthreads(128, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID, uint gi : SV_GroupIndex)
{
    // Whole group, before any barrier. wq:294: 128 threads, 16 clusters of one 8x8.
    if (gid.z >= w[1].w) return;
    {
        uint4 head = w[0];
        uint4 head2 = w[1];
        uint4 ba = w[2u + gid.z * 2u];
        uint4 bb = w[3u + gid.z * 2u];
        resolution = int2(asint(head.x), asint(head.y));
        resolution_8x8_blocks = int2(asint(head.z), asint(head.w));
        coeff_pitch = head2.z;
        input_layer = asint(ba.x);
        quant_resolution = asfloat(ba.y);
        block_offset = asint(ba.z);
        block_stride = asint(ba.w);
        rdo_distortion_scale = asfloat(bb.x);
    }

    for (int zi = int(gi); zi < 2048; zi += 128) {
        gs_staging[zi] = 0u;
    }
    if (gi == 0u) {
        gs_overflow[0] = 0;
        gs_payload_counter[0] = 0u;
        [unroll] for (int c = 0; c < 16; c++) gs_ctotal[c] = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    // Thread mapping matches wavelet_quant.comp main: cluster of 8 lanes,
    // block_x = cluster & 3, block_y = cluster >> 2, group covers 32x32.
    uint cluster = gi >> 3;
    uint block_x = cluster & 3u;
    uint block_y = cluster >> 2;
    uint lane = gi & 7u;
    uint local_coord_x = 4 * (lane >> 2);
    uint local_coord_y = 2 * (lane & 3);
    int2 block8 = int2(int(gid.x) * 4 + int(block_x), int(gid.y) * 4 + int(block_y));
    int2 coord = int2(block8.x * 8 + int(local_coord_x), block8.y * 8 + int(local_coord_y));

    bool in_bounds = (block8.x < resolution_8x8_blocks.x) && (block8.y < resolution_8x8_blocks.y);
    uint block_linear = uint(block_offset) + uint(block_stride) * uint(block8.y) + uint(block8.x);
    int layer = input_layer;

    // ------------------------------------------------------------------
    // P0: Load texels from the DWT coefficient buffer. Texture2D.Load on the
    // HD 6670 disagrees with the UAV store once x or y passes 7.
    // 4x2 pixel order is 0 2 4 6 / 1 3 5 7: (x,y), (x,y+1), (x+1,y), (x+1,y+1).
    // ------------------------------------------------------------------
    int cx = int(coord.x);
    int cy = int(coord.y);

    float t0a_x = band_at(layer, cx, cy);
    float t0a_y = band_at(layer, cx, cy + 1);
    float t0a_z = band_at(layer, cx + 1, cy);
    float t0a_w = band_at(layer, cx + 1, cy + 1);
    precise float4 t0a = float4(t0a_x, t0a_y, t0a_z, t0a_w);

    float t0b_x = band_at(layer, cx + 2, cy);
    float t0b_y = band_at(layer, cx + 2, cy + 1);
    float t0b_z = band_at(layer, cx + 3, cy);
    float t0b_w = band_at(layer, cx + 3, cy + 1);
    precise float4 t0b = float4(t0b_x, t0b_y, t0b_z, t0b_w);

    precise float4 s0 = t0a * quant_resolution;
    precise float4 s1 = t0b * quant_resolution;

    gs_tex[gi][0] = s0.x; gs_tex[gi][1] = s0.y;
    gs_tex[gi][2] = s0.z; gs_tex[gi][3] = s0.w;
    gs_tex[gi][4] = s1.x; gs_tex[gi][5] = s1.y;
    gs_tex[gi][6] = s1.z; gs_tex[gi][7] = s1.w;

    float m_sub = max(max(abs(s0.x), abs(s0.y)), max(abs(s0.z), abs(s0.w)));
    m_sub = max(m_sub, max(max(abs(s1.x), abs(s1.y)), max(abs(s1.z), abs(s1.w))));
    gs_max[gi] = m_sub;

    // Barrier 1
    GroupMemoryBarrierWithGroupSync();

    // ------------------------------------------------------------------
    // P1: m_wave, code, qs, scaled s, ints, msb, active, early(0)
    // ------------------------------------------------------------------
    // Cluster max across all 8 lanes. The old xor only read four originals,
    // so lanes 3 and 7 of a partial 8x8 (the zero 4x2s that do not touch a
    // live lane) kept m_wave of 0. They reserved no payload, the code word
    // still claimed the cluster quality planes, and pack read the next 4x2
    // as this one. That is the scrambled tail of a 512x288 frame.
    uint max_base = gi & ~7u;
    float m_wave = gs_max[max_base];
    [unroll] for (uint mi = 1u; mi < 8u; mi++) m_wave = max(m_wave, gs_max[max_base + mi]);

    uint quant_code = 6;
    float quant_scale = 1.0f;

    if (in_bounds && m_wave >= 1.0f) {
        float y = m_wave - 0.25f;
        uint y_bits = asuint(y);
        int e = int((y_bits >> 23) & 0xffu) - 126;
        float target = float(int(1) << e) - 0.25f;
        float inv = m_wave / target;
        quant_code = uint(ceil((inv - 0.25f) * 8.0f));
        if (quant_code > 15) quant_code = 15;
        quant_scale = asfloat(fs_q2_qs_bits[quant_code]);
    }

    precise float4 s0s = s0 * quant_scale;
    precise float4 s1s = s1 * quant_scale;

    gs_tex[gi][0] = s0s.x; gs_tex[gi][1] = s0s.y;
    gs_tex[gi][2] = s0s.z; gs_tex[gi][3] = s0s.w;
    gs_tex[gi][4] = s1s.x; gs_tex[gi][5] = s1s.y;
    gs_tex[gi][6] = s1s.z; gs_tex[gi][7] = s1s.w;

    int4 a0 = abs(int4(s0s));
    int4 a1 = abs(int4(s1s));

    int max_abs = int(m_wave * quant_scale);
    // rdo_distortion_scale stays in the cbuffer. Quant 0 does not read the error table.
    if (rdo_distortion_scale < -1.0e30f) max_abs = 0;
    int b4x2 = int(m_sub * quant_scale);
    gs_b4x2[gi] = b4x2;

    bool active = in_bounds && (max_abs != 0);
    gs_early[gi] = 0;

    if (lane == 0u && in_bounds && !active) {
        u0.Store(int(block_linear * 8u), 0u);
        u0.Store(int(block_linear * 8u + 4u), 0u);
        u1.Store(int(block_linear * 64u), 0u);
        u1.Store(int(block_linear * 64u + 4u), 0u);
    }

    int msb = active ? msb_of(uint(max_abs)) : -1;

    // Compute early(0) for this thread (q=0)
    int early_t = 0;
    int qp_t = 0;
    int sh_t = b4x2;

    if (active) {
        int block4x2_max = b4x2;
        int msb_q = msb;
        int quality_planes = 0;
        if (msb_q >= 3) {
            quality_planes = msb_q - 2;
            block4x2_max >>= quality_planes;
        }
        // Match wavelet_quant.comp: once quality planes exist, the sign byte
        // replaces the (block>0) bit. Adding quality_planes on top drops that
        // byte for lanes whose own max shifts to zero, and the next lane's
        // payload overlaps.
        int encode_cost_early = (block4x2_max > 0) ? 1 : 0;
        if (msb_q >= 3) {
            encode_cost_early = quality_planes + 1;
        }
        encode_cost_early += msb_of(uint(block4x2_max)) + 1;
        early_t = encode_cost_early;
        qp_t = quality_planes;
        sh_t = block4x2_max;
    }

    gs_early[gi] = early_t;

    // Barrier 2
    GroupMemoryBarrierWithGroupSync();

    // Exclusive prefix of the 8 lane costs. The old xor scan mixed lanes 2 and 3.
    int scan_excl = 0;
    int cluster_sum = 0;
    {
        uint base_lane = gi & ~7u;
        [unroll] for (uint sum_i = 0u; sum_i < 8u; sum_i++) cluster_sum += gs_early[base_lane + sum_i];
        [unroll] for (uint pref_i = 0u; pref_i < lane; pref_i++) scan_excl += gs_early[base_lane + pref_i];
    }
    if (lane == 7u) gs_ctotal[cluster] = uint(cluster_sum);
    GroupMemoryBarrierWithGroupSync();

    // ------------------------------------------------------------------
    // P3: Sum totals, round up to 4-byte, allocate payload bytes at u2 offset 0
    // Use groupshared counter since RWByteAddressBuffer doesn't support
    // InterlockedAdd in fxc cs_5_0 — we write to u2 after the barrier.
    // ------------------------------------------------------------------
    int group_base = 0;

    if (gi == 0u) {
        // Each cluster keeps the 4-byte pad the one-8x8 dispatch used.
        uint total = 0u;
        [unroll] for (int c = 0; c < 16; c++) {
            uint t = uint(gs_ctotal[c]);
            if (t > 0u) t = (t + 3u) & ~3u;
            gs_ctotal[c] = int(t);
            total += t;
        }
        uint base_byte = 0u;
        if (total > 0u) u2.InterlockedAdd(0, total, base_byte);
        gs_payload_counter[0] = total;
        gs_group_base[0] = int(base_byte);
    }

    // Barrier 4
    GroupMemoryBarrierWithGroupSync();

    // All threads read the counter and compute group_base
    // In the real implementation, we'd need a way to get the previous counter value.
    // For now, use a simple approach: each group's base = sum of all previous groups' sizes.
    // Since we can't atomically add, we use a separate counter buffer approach.
    // The host code tracks group bases.
    group_base = gs_group_base[0];

    // ------------------------------------------------------------------
    // P4: cluster_base, payload bytes, meta, stats
    // ------------------------------------------------------------------
    int cluster_base = group_base;
    for (uint pc = 0u; pc < cluster; pc++) cluster_base += int(gs_ctotal[pc]);
    if (active) {
        // Write payload bytes
        bool need_sign = (sh_t != 0) || (qp_t != 0);
        if (need_sign && early_t > 0) {
            uint sign_bits = 0u;
            sign_bits |= (s0s.x < 0.0f) ? (1u << 0) : 0u;
            sign_bits |= (s0s.y < 0.0f) ? (1u << 1) : 0u;
            sign_bits |= (s0s.z < 0.0f) ? (1u << 2) : 0u;
            sign_bits |= (s0s.w < 0.0f) ? (1u << 3) : 0u;
            sign_bits |= (s1s.x < 0.0f) ? (1u << 4) : 0u;
            sign_bits |= (s1s.y < 0.0f) ? (1u << 5) : 0u;
            sign_bits |= (s1s.z < 0.0f) ? (1u << 6) : 0u;
            sign_bits |= (s1s.w < 0.0f) ? (1u << 7) : 0u;

            // Staging is per cluster. cluster_base is the global payload byte.
            int byte_offset = scan_excl;
            if (byte_offset >= 0 && byte_offset < 512) {
                uint word_idx = uint(byte_offset) >> 2;
                uint bit_shift = uint((byte_offset & 3) << 3);
                InterlockedOr(gs_staging[cluster * 128u + word_idx], sign_bits << bit_shift);
            }

            int plane_code = msb_of(uint(sh_t)) + 1;
            int plane_iterations = qp_t + plane_code;

            for (int q = plane_iterations - 1; q >= 0; q--) {
                uint plane_bits = 0u;
                plane_bits |= (uint(a0.x >> q) & 1u) << 0;
                plane_bits |= (uint(a0.y >> q) & 1u) << 1;
                plane_bits |= (uint(a0.z >> q) & 1u) << 2;
                plane_bits |= (uint(a0.w >> q) & 1u) << 3;
                plane_bits |= (uint(a1.x >> q) & 1u) << 4;
                plane_bits |= (uint(a1.y >> q) & 1u) << 5;
                plane_bits |= (uint(a1.z >> q) & 1u) << 6;
                plane_bits |= (uint(a1.w >> q) & 1u) << 7;

                byte_offset++;
                if (byte_offset >= 0 && byte_offset < 512) {
                    uint word_idx = uint(byte_offset) >> 2;
                    uint bit_shift = uint((byte_offset & 3) << 3);
                    InterlockedOr(gs_staging[cluster * 128u + word_idx], plane_bits << bit_shift);
                }
            }
        }
    }

    // Each lane publishes its own 2-bit plane code. Lane 0 merges the cluster.
    gs_pcode[gi] = 0u;
    if (active) {
        int pc = msb_of(uint(sh_t)) + 1;
        if (pc < 0) pc = 0;
        gs_pcode[gi] = uint(pc);
    }
    GroupMemoryBarrierWithGroupSync();

    // Lane 0: meta, num_planes, errors[0], errors[msb+1]
    if (lane == 0u && active) {
        uint block_index = block_linear;

        uint code_word = uint(qp_t) << Q_PLANES_OFFSET;
        code_word |= (quant_code << QUANT_SCALE_OFFSET);

        uint merged = 0u;
        [unroll] for (uint lane_i = 0u; lane_i < 8u; lane_i++)
            merged |= gs_pcode[gi + lane_i] << (2u * lane_i);
        code_word |= merged;

        u0.Store(int(block_index * 8u), code_word);
        u0.Store(int(block_index * 8u + 4u), uint(cluster_base));

        // num_sig(0) over this cluster's 8 threads, 8 texels each.
        int wave8_num_significants = 0;
        for (int sig_i = 0; sig_i < 8; sig_i++) {
            int sig = 0;
            for (int kk = 0; kk < 8; kk++) {
                if (abs(gs_tex[gi + uint(sig_i)][kk]) >= 1.0f) sig++;
            }
            wave8_num_significants += sig;
        }

        // encode_cost_late_bits is a clustered sum, not lane 0 alone.
        // Lane 0 only made pack allocate a few words, then the decoder
        // walked off the end of the block into the next header.
        int mag_bytes = 0;
        {
            uint base_lane = gi & ~7u;
            [unroll] for (uint mb = 0u; mb < 8u; mb++)
                mag_bytes += max(gs_early[base_lane + mb] - 1, 0);
        }
        int late_bits_0 = 8 * mag_bytes + wave8_num_significants;
        uint errors_0 = uint(late_bits_0 & 0xffff) << 16;

        uint num_planes = uint(msb + 1);

        if (msb + 1 > 14) {
            InterlockedAdd(gs_overflow[0], 1);
        }

        u1.Store4(int(block_index * 64 + 0), uint4(num_planes, errors_0, 0, 0));
        // Rate control stays at quant 0, so the per-q error table is not read.
    }

    // Barrier 5
    GroupMemoryBarrierWithGroupSync();

    // ------------------------------------------------------------------
    // P5: Copy staging words to u2
    // ------------------------------------------------------------------
    uint my_words = uint(gs_ctotal[cluster]) >> 2;
    uint dest = uint(cluster_base);
    for (uint ci = lane; ci < my_words; ci += 8u) {
        if (ci < 128u)
            u2.Store(int(8u + dest + ci * 4u), gs_staging[cluster * 128u + ci]);
    }

    // Flush
    GroupMemoryBarrierWithGroupSync();
}
