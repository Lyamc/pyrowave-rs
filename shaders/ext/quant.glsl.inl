/* OpenGL 4.3 port of pyrowave-dx11 quant.hlsl (wavelet_quant.comp, one 8x8 per group).
 * FS-Q2 scale bits are the CPU reference from probe_fs_q2.inc.
 */
static const char *kQuantComp = R"GL(#version 430
layout(local_size_x = 8) in;

struct BandQ {
    float quant_resolution;
    int block_offset;
    int layer;
    int _pad;
};

layout(std140, binding = 0) uniform Push {
    ivec2 resolution;
    ivec2 resolution_8x8_blocks;
    vec2 inv_resolution;
    int block_stride;
    float rdo_distortion_scale;
    uint coeff_pitch;
    int band_count;
    ivec2 _pad_align;
    BandQ bands[4];
};

layout(std430, binding = 1) readonly buffer CoeffIn { float coeff[]; } coeff_in;
layout(std430, binding = 2) buffer Meta { uint u0[]; };
layout(std430, binding = 3) buffer Stats { uint u1[]; };
layout(std430, binding = 4) buffer Payload { uint u2[]; };

const uint Q_PLANES_OFFSET = 16u;
const uint QUANT_SCALE_OFFSET = 20u;

const uint fs_q2_qs_bits[16] = uint[16](
    0x40800000u, 0x402aaaabu, 0x40000000u, 0x3fcccccdu,
    0x3faaaaabu, 0x3f924925u, 0x3f800000u, 0x3f638e39u,
    0x3f4ccccdu, 0x3f3a2e8cu, 0x3f2aaaabu, 0x3f1d89d9u,
    0x3f124925u, 0x3f088889u, 0x3f000000u, 0x3ef0f0f1u
);

shared float gs_max[8];
shared float gs_tex[8][8];
shared int gs_b4x2[8];
shared int gs_early[8];
shared int gs_ctotal[16];
shared int gs_group_base;
shared uint gs_staging[640];
shared int gs_overflow;
shared uint gs_payload_counter;
shared uint gs_pcode[8];

int msb_of(uint v)
{
    if (v == 0u) return -1;
    return findMSB(v);
}

float make_pow2(int q)
{
    return uintBitsToFloat(uint((127 + q) << 23));
}

float band_at(int layer, int x, int y)
{
    if (uint(x) >= uint(resolution.x) || uint(y) >= uint(resolution.y)) return 0.0;
    uint pitch_h = (uint(resolution.y) + 31u) & ~31u;
    uint idx = uint(layer) * coeff_pitch * pitch_h + uint(y) * coeff_pitch + uint(x);
    // A direct `return coeff[idx]` is error #247 on the HD 6670 compiler.
    float v = coeff_in.coeff[idx];
    return v;
}

void store_u32(uint byte_off, uint v)
{
    u0[byte_off >> 2] = v;
}

void store_stats4(uint byte_off, uvec4 v)
{
    uint i = byte_off >> 2;
    u1[i] = v.x;
    u1[i + 1u] = v.y;
    u1[i + 2u] = v.z;
    u1[i + 3u] = v.w;
}

void main()
{
    int band_i = int(gl_WorkGroupID.z);
    float quant_resolution = bands[band_i].quant_resolution;
    int block_offset = bands[band_i].block_offset;
    int input_layer = bands[band_i].layer;
    uint gi = gl_LocalInvocationIndex;
    for (int zi = int(gi); zi < 640; zi += 8) gs_staging[zi] = 0u;
    if (gi == 0u) {
        gs_overflow = 0;
        gs_payload_counter = 0u;
        for (int c = 0; c < 16; c++) gs_ctotal[c] = 0;
    }
    barrier();

    uint lane = gi;
    uint local_coord_x = 4u * (lane >> 2);
    uint local_coord_y = 2u * (lane & 3u);
    ivec2 coord = ivec2(int(gl_WorkGroupID.x) * 8 + int(local_coord_x),
                        int(gl_WorkGroupID.y) * 8 + int(local_coord_y));
    ivec2 block8 = ivec2(gl_WorkGroupID.xy);
    bool in_bounds = (block8.x < resolution_8x8_blocks.x) && (block8.y < resolution_8x8_blocks.y);
    uint block_linear = uint(block_offset) + uint(block_stride) * uint(block8.y) + uint(block8.x);
    int layer = int(input_layer);

    int cx = coord.x;
    int cy = coord.y;
    precise vec4 s0 = vec4(
        band_at(layer, cx, cy),
        band_at(layer, cx, cy + 1),
        band_at(layer, cx + 1, cy),
        band_at(layer, cx + 1, cy + 1)) * quant_resolution;
    precise vec4 s1 = vec4(
        band_at(layer, cx + 2, cy),
        band_at(layer, cx + 2, cy + 1),
        band_at(layer, cx + 3, cy),
        band_at(layer, cx + 3, cy + 1)) * quant_resolution;

    gs_tex[gi][0] = s0.x; gs_tex[gi][1] = s0.y;
    gs_tex[gi][2] = s0.z; gs_tex[gi][3] = s0.w;
    gs_tex[gi][4] = s1.x; gs_tex[gi][5] = s1.y;
    gs_tex[gi][6] = s1.z; gs_tex[gi][7] = s1.w;

    float m_sub = max(max(abs(s0.x), abs(s0.y)), max(abs(s0.z), abs(s0.w)));
    m_sub = max(m_sub, max(max(abs(s1.x), abs(s1.y)), max(abs(s1.z), abs(s1.w))));
    gs_max[gi] = m_sub;
    barrier();

    float m_wave = gs_max[0];
    m_wave = max(m_wave, gs_max[1]);
    m_wave = max(m_wave, gs_max[2]);
    m_wave = max(m_wave, gs_max[3]);
    m_wave = max(m_wave, gs_max[4]);
    m_wave = max(m_wave, gs_max[5]);
    m_wave = max(m_wave, gs_max[6]);
    m_wave = max(m_wave, gs_max[7]);

    uint quant_code = 6u;
    float quant_scale = 1.0;
    if (in_bounds && m_wave >= 1.0) {
        float y = m_wave - 0.25;
        uint y_bits = floatBitsToUint(y);
        int e = int((y_bits >> 23) & 0xffu) - 126;
        float target = float(int(1) << e) - 0.25;
        float inv = m_wave / target;
        quant_code = uint(ceil((inv - 0.25) * 8.0));
        if (quant_code > 15u) quant_code = 15u;
        quant_scale = uintBitsToFloat(fs_q2_qs_bits[quant_code]);
    }

    precise vec4 s0s = s0 * quant_scale;
    precise vec4 s1s = s1 * quant_scale;
    gs_tex[gi][0] = s0s.x; gs_tex[gi][1] = s0s.y;
    gs_tex[gi][2] = s0s.z; gs_tex[gi][3] = s0s.w;
    gs_tex[gi][4] = s1s.x; gs_tex[gi][5] = s1s.y;
    gs_tex[gi][6] = s1s.z; gs_tex[gi][7] = s1s.w;

    ivec4 a0 = abs(ivec4(s0s));
    ivec4 a1 = abs(ivec4(s1s));
    int max_abs = int(m_wave * quant_scale);
    int b4x2 = int(m_sub * quant_scale);
    gs_b4x2[gi] = b4x2;

    bool live = in_bounds && (max_abs != 0);
    gs_early[gi] = 0;
    if (lane == 0u && in_bounds && !live) {
        store_u32(block_linear * 8u, 0u);
        store_u32(block_linear * 8u + 4u, 0u);
    }

    int msb = live ? msb_of(uint(max_abs)) : -1;
    int early_t = 0;
    int qp_t = 0;
    int sh_t = b4x2;
    if (live) {
        int block4x2_max = b4x2;
        int msb_q = msb;
        int quality_planes = 0;
        if (msb_q >= 3) {
            quality_planes = msb_q - 2;
            block4x2_max >>= quality_planes;
        }
        int encode_cost_early = (block4x2_max > 0) ? 1 : 0;
        if (msb_q >= 3) encode_cost_early = quality_planes + 1;
        encode_cost_early += msb_of(uint(block4x2_max)) + 1;
        early_t = encode_cost_early;
        qp_t = quality_planes;
        sh_t = block4x2_max;
    }
    gs_early[gi] = early_t;
    barrier();

    int scan_excl = 0;
    int cluster_sum = 0;
    {
        uint base_lane = gi & ~7u;
        for (uint sum_i = 0u; sum_i < 8u; sum_i++) cluster_sum += gs_early[base_lane + sum_i];
        for (uint pref_i = 0u; pref_i < lane; pref_i++) scan_excl += gs_early[base_lane + pref_i];
    }
    if (lane == 7u) gs_ctotal[0] = cluster_sum;
    barrier();

    if (gi == 0u) {
        uint total = uint(gs_ctotal[0]);
        if (total > 0u) total = (total + 3u) & ~3u;
        uint base_byte = 0u;
        if (total > 0u) base_byte = atomicAdd(u2[0], total);
        gs_payload_counter = total;
        gs_group_base = int(base_byte);
    }
    barrier();
    int group_base = gs_group_base;

    int cluster_base = group_base;
    if (live) {
        bool need_sign = (sh_t != 0) || (qp_t != 0);
        if (need_sign && early_t > 0) {
            uint sign_bits = 0u;
            sign_bits |= (s0s.x < 0.0) ? (1u << 0) : 0u;
            sign_bits |= (s0s.y < 0.0) ? (1u << 1) : 0u;
            sign_bits |= (s0s.z < 0.0) ? (1u << 2) : 0u;
            sign_bits |= (s0s.w < 0.0) ? (1u << 3) : 0u;
            sign_bits |= (s1s.x < 0.0) ? (1u << 4) : 0u;
            sign_bits |= (s1s.y < 0.0) ? (1u << 5) : 0u;
            sign_bits |= (s1s.z < 0.0) ? (1u << 6) : 0u;
            sign_bits |= (s1s.w < 0.0) ? (1u << 7) : 0u;

            int byte_offset = cluster_base - group_base + scan_excl;
            if (byte_offset >= 0 && byte_offset < 2560) {
                uint word_idx = uint(byte_offset) >> 2;
                uint bit_shift = uint((byte_offset & 3) << 3);
                atomicOr(gs_staging[word_idx], sign_bits << bit_shift);
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
                if (byte_offset >= 0 && byte_offset < 2560) {
                    uint word_idx = uint(byte_offset) >> 2;
                    uint bit_shift = uint((byte_offset & 3) << 3);
                    atomicOr(gs_staging[word_idx], plane_bits << bit_shift);
                }
            }
        }
    }

    gs_pcode[gi] = 0u;
    if (live) {
        int pc = msb_of(uint(sh_t)) + 1;
        if (pc < 0) pc = 0;
        gs_pcode[gi] = uint(pc);
    }
    barrier();

    if (lane == 0u && live) {
        uint code_word = uint(qp_t) << Q_PLANES_OFFSET;
        code_word |= (quant_code << QUANT_SCALE_OFFSET);
        uint merged = 0u;
        for (uint lane_i = 0u; lane_i < 8u; lane_i++)
            merged |= gs_pcode[gi + lane_i] << (2u * lane_i);
        code_word |= merged;
        store_u32(block_linear * 8u, code_word);
        store_u32(block_linear * 8u + 4u, uint(cluster_base));
    }
    barrier();

    int my_words = int(gs_payload_counter) >> 2;
    for (int ci = int(gi); ci < my_words; ci += 8) {
        if (ci < 640) u2[(8u + uint(group_base) + uint(ci) * 4u) >> 2] = gs_staging[ci];
    }
    barrier();
}
)GL";
