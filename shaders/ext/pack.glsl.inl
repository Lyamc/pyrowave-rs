/* OpenGL 4.3 port of pyrowave-dx11 pack.hlsl (block_packing.comp).
 * One 64-thread group packs a 2x2 of 32x32 blocks. Quant stays 0 unless the
 * quant buffer says otherwise. Packet bytes are staged in shared memory and
 * stored as whole words.
 */
static const char *kPackComp = R"GL(#version 430
layout(local_size_x = 64) in;

struct BandP {
    uint quant_code;
    int off32;
    int off8;
    int _pad;
};

layout(std140, binding = 0) uniform Push {
    ivec2 resolution;
    ivec2 resolution_32x32_blocks;
    ivec2 resolution_8x8_blocks;
    uint sequence_code;
    int block_stride_32x32;
    int block_stride_8x8;
    int band_count;
    ivec2 _pad_align;
    BandP bands[4];
};

layout(std430, binding = 1) buffer Bitstream { uint u0[]; };
layout(std430, binding = 2) buffer MetaOut { uint u1[]; };
layout(std430, binding = 3) readonly buffer BlockMeta { uint t0[]; };
layout(std430, binding = 4) buffer Payloads { uint u2[]; };
layout(std430, binding = 5) readonly buffer Quants { int t2[]; };

const uint Q_PLANES_OFFSET = 16u;
const uint MAX_PACKET_WORDS = 640u;

shared uint gs_pkt[4 * 640];
shared uint gs_ballot[4];
shared uint gs_plane_bytes[64];
shared uint gs_sign_bits[64];
shared uint gs_bits_meta[64];
shared uint gs_base[4];
shared uint gs_sign[4][32];
shared uint gs_overflow;

uint compute_required_8x8_size(uint control_word)
{
    int q_bits = int((control_word >> Q_PLANES_OFFSET) & 0xfu);
    uint lsbs = control_word & 0x5555u;
    uint msbs = control_word & 0xaaaau;
    msbs |= msbs >> 1;
    return uint(bitCount(lsbs) + bitCount(msbs)) + uint(q_bits) * 8u;
}

uint quantize_code_word(uint control_word, int quant)
{
    if (quant != 0 && control_word != 0u) {
        int q_bits = int((control_word >> Q_PLANES_OFFSET) & 0xfu);
        int sub_quant = min(q_bits, quant);
        q_bits -= sub_quant;
        quant -= sub_quant;
        if (quant != 0) {
            quant = min(quant, 3);
            uint plane0 = control_word & 0x5555u;
            uint plane1 = (control_word & 0xaaaau) >> 1;
            uint plane2 = plane0 & plane1;
            while (quant != 0) {
                plane0 = plane1;
                plane1 = plane2;
                plane2 = 0u;
                quant--;
            }
            plane0 &= ~plane1;
            uint new_control_word = plane0 | (plane1 << 1);
            control_word = (control_word & ~0xffffu) | (new_control_word & 0xffffu);
        }
        control_word = (control_word & ~0xf0000u) | (uint(q_bits) << Q_PLANES_OFFSET);
    }
    return control_word;
}

uint modify_quant_code(uint code, int quant)
{
    int e = int((code >> 3) & 0x1fu);
    e = max(e - quant, 0);
    return (code & ~0x3f8u) | (uint(e) << 3);
}

uint load_payload_byte(uint idx)
{
    uint addr = idx + 8u;
    uint word = u2[addr >> 2];
    return (word >> (8u * (addr & 3u))) & 0xffu;
}

void pkt_put_byte(uint blk, uint local_byte, uint value)
{
    uint word_idx = local_byte >> 2u;
    uint shift = (local_byte & 3u) << 3u;
    uint idx = blk * MAX_PACKET_WORDS + word_idx;
    uint mask = 0xffu << shift;
    atomicAnd(gs_pkt[idx], ~mask);
    atomicOr(gs_pkt[idx], (value & 0xffu) << shift);
}

void main()
{
    int band_i = int(gl_WorkGroupID.z);
    uint quant_resolution_code = bands[band_i].quant_code;
    int block_offset_32x32 = bands[band_i].off32;
    int block_offset_8x8 = bands[band_i].off8;
    int resolution_x = resolution.x;
    int resolution_y = resolution.y;
    int resolution_32x32_blocks_x = resolution_32x32_blocks.x;
    int resolution_32x32_blocks_y = resolution_32x32_blocks.y;
    int resolution_8x8_blocks_x = resolution_8x8_blocks.x;
    int resolution_8x8_blocks_y = resolution_8x8_blocks.y;
    uint gi = gl_LocalInvocationIndex;
    uint blk = gi >> 4;
    for (uint w = gi; w < 4u * MAX_PACKET_WORDS; w += 64u) gs_pkt[w] = 0u;
    if (gi < 4u) {
        gs_ballot[gi] = 0u;
        gs_base[gi] = 0u;
    }
    for (uint s = gi; s < 128u; s += 64u) gs_sign[s >> 5u][s & 31u] = 0u;
    if (gi == 0u) gs_overflow = 0u;
    barrier();

    uint local_block_index = gi & 15u;
    uint local_x = local_block_index & 3u;
    uint local_y = local_block_index >> 2;

    int block32x32_x = 2 * int(gl_WorkGroupID.x) + int((gi >> 4) & 1u);
    int block32x32_y = 2 * int(gl_WorkGroupID.y) + int((gi >> 5) & 1u);
    int block8x8_x = 4 * block32x32_x + int(local_x);
    int block8x8_y = 4 * block32x32_y + int(local_y);

    bool in_range_8x8 = (block8x8_x < resolution_8x8_blocks_x) && (block8x8_y < resolution_8x8_blocks_y);
    bool in_range_32x32 = (block32x32_x < resolution_32x32_blocks_x) && (block32x32_y < resolution_32x32_blocks_y);

    int quant = 0;
    if (in_range_32x32) {
        int bo32 = block_offset_32x32 + block_stride_32x32 * block32x32_y + block32x32_x;
        quant = t2[bo32];
    }

    uint code_word = 0u;
    if (in_range_8x8) {
        int b8 = block_offset_8x8 + block_stride_8x8 * block8x8_y + block8x8_x;
        code_word = t0[uint(b8) * 2u];
    }
    code_word = quantize_code_word(code_word, quant);
    bool live = (code_word & 0xffffu) != 0u;
    if (live) atomicOr(gs_ballot[blk], 1u << local_block_index);

    uint plane_bytes = compute_required_8x8_size(code_word);
    uint bits_meta = live ? (plane_bytes * 8u + 64u + 24u) : 0u;
    gs_plane_bytes[gi] = plane_bytes;
    gs_sign_bits[gi] = 0u;
    gs_bits_meta[gi] = bits_meta;
    barrier();

    uint block_bits_meta = 0u;
    for (uint j = 0u; j < 16u; j++) block_bits_meta += gs_bits_meta[blk * 16u + j];
    uint total_words = (block_bits_meta + 31u) / 32u;
    if (total_words != 0u) total_words += 2u;

    bool writes_header = in_range_32x32 && (local_block_index == 15u);
    uint base = 0u;
    if (writes_header && total_words != 0u) base = atomicAdd(u2[1], total_words);
    if (local_block_index == 15u) gs_base[blk] = base;
    barrier();
    base = gs_base[blk];

    uint exclusive_plane = 0u;
    for (uint jj = 0u; jj < local_block_index; jj++)
        exclusive_plane += gs_plane_bytes[blk * 16u + jj];
    uint total_plane_bytes = 0u;
    for (uint mm = 0u; mm < 16u; mm++)
        total_plane_bytes += gs_plane_bytes[blk * 16u + mm];
    uint N = uint(bitCount(gs_ballot[blk]));

    if (writes_header && total_words != 0u) {
        uint ballot_val = gs_ballot[blk] & 0xffffu;
        gs_pkt[blk * MAX_PACKET_WORDS + 0u] = ballot_val | (total_words << 16) | (sequence_code << 28);
        int block_index = block_offset_32x32 + block_stride_32x32 * block32x32_y + block32x32_x;
        gs_pkt[blk * MAX_PACKET_WORDS + 1u] = modify_quant_code(quant_resolution_code, quant) | (uint(block_index) << 8);
        if (total_words <= MAX_PACKET_WORDS) {
            u1[uint(block_index) * 2u] = base;
            u1[uint(block_index) * 2u + 1u] = total_words;
        }
    }

    uint local_sign_bits = 0u;
    if (live && total_words != 0u && total_words <= MAX_PACKET_WORDS) {
        uint hdr_offset = uint(bitCount(gs_ballot[blk] & ((1u << local_block_index) - 1u)));
        uint cw_byte = 8u + 2u * hdr_offset;
        pkt_put_byte(blk, cw_byte, code_word & 0xffu);
        pkt_put_byte(blk, cw_byte + 1u, (code_word >> 8u) & 0xffu);
        uint q_byte_addr = 8u + 2u * N + hdr_offset;
        pkt_put_byte(blk, q_byte_addr, (code_word >> 16u) & 0xffu);

        uint meta_addr_words = uint(block_offset_8x8 + block_stride_8x8 * block8x8_y + block8x8_x) * 2u;
        uint meta_cw = t0[meta_addr_words];
        uint in_q_bits = (meta_cw >> Q_PLANES_OFFSET) & 0xfu;
        uint out_q_bits = (code_word >> Q_PLANES_OFFSET) & 0xfu;
        uint in_offset = t0[meta_addr_words + 1u];
        uint out_offset = 8u + 3u * N + exclusive_plane;

        for (int sb = 0; sb < 16; sb += 2) {
            uint out_planes = ((code_word >> uint(sb)) & 3u) + out_q_bits;
            uint in_planes = ((meta_cw >> uint(sb)) & 3u) + in_q_bits;
            if (in_planes != 0u) in_planes++;
            if (in_planes != 0u) {
                uint sign_plane = load_payload_byte(in_offset);
                uint significant_mask = 0u;
                for (uint p = 0u; p < out_planes; p++) {
                    uint byte_val = load_payload_byte(in_offset + 1u + p);
                    significant_mask |= byte_val;
                    pkt_put_byte(blk, out_offset + p, byte_val);
                }
                out_offset += out_planes;
                uint sig_mask = significant_mask;
                while (sig_mask != 0u) {
                    uint coef = uint(findLSB(sig_mask));
                    sig_mask &= sig_mask - 1u;
                    uint sign_bit = (sign_plane >> coef) & 1u;
                    uint sword = (local_block_index << 1) + (local_sign_bits >> 5u);
                    uint sshift = local_sign_bits & 31u;
                    if (sword < 32u) atomicOr(gs_sign[blk][sword], sign_bit << sshift);
                    local_sign_bits++;
                }
            }
            in_offset += in_planes;
        }
    }
    gs_sign_bits[gi] = local_sign_bits;
    barrier();

    if (in_range_32x32 && total_words != 0u && total_words <= MAX_PACKET_WORDS && local_block_index == 0u) {
        uint sign_base = 8u + 3u * N + total_plane_bytes;
        uint acc = 0u;
        uint nacc = 0u;
        uint out_i = 0u;
        for (uint lane = 0u; lane < 16u; lane++) {
            uint nbits = gs_sign_bits[blk * 16u + lane];
            uint base_word = lane << 1;
            for (uint b = 0u; b < nbits; b++) {
                uint src = gs_sign[blk][base_word + (b >> 5u)];
                uint bit = (src >> (b & 31u)) & 1u;
                acc |= bit << nacc;
                nacc++;
                if (nacc == 8u) {
                    pkt_put_byte(blk, sign_base + out_i, acc);
                    out_i++;
                    acc = 0u;
                    nacc = 0u;
                }
            }
        }
        if (nacc != 0u) pkt_put_byte(blk, sign_base + out_i, acc);
    }
    barrier();

    if (in_range_32x32 && total_words != 0u && total_words <= MAX_PACKET_WORDS) {
        for (uint w = local_block_index; w < total_words; w += 16u) {
            u0[base + w] = gs_pkt[blk * MAX_PACKET_WORDS + w];
        }
    }
}
)GL";
