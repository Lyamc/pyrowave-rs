// pack.hlsl -- Block packing: produce bitstream bytes
// Port of R/shaders/block_packing.comp to cs_5_0
//
// Spec: docs/stages/05_block_packing.md
// Source: R/shaders/block_packing.comp

#include "common.hlsli"

// ------------------------------------------------------------------
// Constants
// ------------------------------------------------------------------
#define Q_PLANES_OFFSET 16
#define Q_PLANES_BITS 4
#define MAX_PACKET_WORDS 640  // Max words per 32x32 block

// ------------------------------------------------------------------
// One job per band, 48 bytes. GroupID.z selects the band so every band
// is a slice of a single dispatch. Matches BlockPackingPushData.
// ------------------------------------------------------------------
ByteAddressBuffer tJobs : register(t3);
static int resolution_x;
static int resolution_y;
static int resolution_32x32_blocks_x;
static int resolution_32x32_blocks_y;
static int resolution_8x8_blocks_x;
static int resolution_8x8_blocks_y;
static uint quant_resolution_code;
static uint sequence_code;
static int block_offset_32x32;
static int block_stride_32x32;
static int block_offset_8x8;
static int block_stride_8x8;

void load_job(uint z)
{
    uint b = z * 48u;
    resolution_x = asint(tJobs.Load(b + 0u));
    resolution_y = asint(tJobs.Load(b + 4u));
    resolution_32x32_blocks_x = asint(tJobs.Load(b + 8u));
    resolution_32x32_blocks_y = asint(tJobs.Load(b + 12u));
    resolution_8x8_blocks_x = asint(tJobs.Load(b + 16u));
    resolution_8x8_blocks_y = asint(tJobs.Load(b + 20u));
    quant_resolution_code = tJobs.Load(b + 24u);
    sequence_code = tJobs.Load(b + 28u);
    block_offset_32x32 = asint(tJobs.Load(b + 32u));
    block_stride_32x32 = asint(tJobs.Load(b + 36u));
    block_offset_8x8 = asint(tJobs.Load(b + 40u));
    block_stride_8x8 = asint(tJobs.Load(b + 44u));
}

// ------------------------------------------------------------------
// Resources
// ------------------------------------------------------------------
// u0: RWByteAddressBuffer = bitstream (packets)
RWByteAddressBuffer u0 : register(u0);

// u1: RWByteAddressBuffer = bitstream meta (BitstreamPacket per block)
RWByteAddressBuffer u1 : register(u1);

// t0: ByteAddressBuffer = block meta (stage 2)
ByteAddressBuffer t0 : register(t0);

// u2: RWByteAddressBuffer = payloads (counter @4, data @8)
RWByteAddressBuffer u2 : register(u2);

// t1: ByteAddressBuffer = block stats (stage 2)
ByteAddressBuffer t1 : register(t1);

// t2: ByteAddressBuffer = quant buffer (stage 4)
ByteAddressBuffer t2 : register(t2);

// ------------------------------------------------------------------
// Groupshared memory (global scope)
// ------------------------------------------------------------------
// gs_pkt: staging packet per 32x32 block. 4 blocks x 640 words = 2560 uints = 10 KB
groupshared uint gs_pkt[4 * MAX_PACKET_WORDS];

// gs_ballot: 4 uints per 32x32 block (16 bits each, stored in lower 16 bits)
groupshared uint gs_ballot[4];

// Per-lane data (64 threads x 16 lanes per block)
groupshared uint gs_plane_bytes[64];
groupshared uint gs_sign_bits[64];
groupshared uint gs_bits_meta[64];

// gs_base: broadcast base from lane 15
groupshared uint gs_base[4];

// Sign bits, 64 per 8x8 lane (2 uints). Compacted after the copy.
groupshared uint gs_sign[4][32];

// Debug overflow counter
groupshared uint gs_overflow[1];

// ------------------------------------------------------------------
// Helper: compute_required_8x8_size (bp:97-105)
// ------------------------------------------------------------------
uint compute_required_8x8_size(uint control_word)
{
    int q_bits = int((control_word >> Q_PLANES_OFFSET) & 0xfu);
    uint lsbs = control_word & 0x5555u;
    uint msbs = control_word & 0xaaaau;
    uint msbs_shift = msbs >> 1;
    msbs |= msbs_shift;
    return countbits(lsbs) + countbits(msbs) + uint(q_bits) * 8u;
}

// ------------------------------------------------------------------
// Helper: quantize_code_word (bp:107-142)
// ------------------------------------------------------------------
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

// ------------------------------------------------------------------
// Helper: modify_quant_code (bp:161-167)
// ------------------------------------------------------------------
uint modify_quant_code(uint code, int quant)
{
    int e = int((code >> 3) & 0x1fu);
    e = max(e - quant, 0);
    return (code & ~0x3f8u) | (uint(e) << 3);
}

// ------------------------------------------------------------------
// Helper: load byte from payload buffer (bp:340)
// ------------------------------------------------------------------
uint load_payload_byte(RWByteAddressBuffer payload, uint idx)
{
    uint word = payload.Load((idx + 8u) & ~3u);
    return (word >> (8u * ((idx + 8u) & 3u))) & 0xffu;
}

// Packet staging is local to the 32x32 block. gs_pkt is zeroed, and each
// byte is written once, so one OR publishes it. A whole word is one atomic.
uint pkt_low_mask(uint n)
{
    if (n >= 4u) return 0xffffffffu;
    return (1u << (n * 8u)) - 1u;
}

void pkt_or_bytes(uint blk, uint local_byte, uint value, uint n)
{
    if (n == 0u) return;
    uint word_idx = local_byte >> 2u;
    uint shift = (local_byte & 3u) << 3u;
    uint idx = blk * MAX_PACKET_WORDS + word_idx;
    InterlockedOr(gs_pkt[idx], (value & pkt_low_mask(n)) << shift);
}

void pkt_put_byte(uint blk, uint local_byte, uint value)
{
    pkt_or_bytes(blk, local_byte, value, 1u);
}

// ------------------------------------------------------------------
// Main
// ------------------------------------------------------------------
[numthreads(64, 1, 1)]
void main(uint3 grp : SV_GroupID, uint3 gtid : SV_GroupThreadID, uint gi : SV_GroupIndex)
{
    load_job(grp.z);
    // This group covers a 2x2 of 32x32 blocks. Groups past the band exit
    // before any barrier; every thread in the group takes the same path.
    if ((int(grp.x) * 2) >= resolution_32x32_blocks_x || (int(grp.y) * 2) >= resolution_32x32_blocks_y)
        return;

    // ------------------------------------------------------------------
    // P0: Zero staging arrays and ballot
    // ------------------------------------------------------------------
    // Zero every staging word. The old loop only cleared 16 words per block,
    // and it used the thread id as the word index, so blocks 1..3 lost their headers.
    uint blk = gi >> 4;
    for (uint w = gi; w < 4u * MAX_PACKET_WORDS; w += 64u) {
        gs_pkt[w] = 0u;
    }
    if (gi < 4u) {
        gs_ballot[gi] = 0u;
        gs_base[gi] = 0u;
    }
    for (uint s = gi; s < 128u; s += 64u) gs_sign[s >> 5u][s & 31u] = 0u;
    if (gi == 0u) gs_overflow[0] = 0u;

    GroupMemoryBarrierWithGroupSync();  // B1

    // ------------------------------------------------------------------
    // P1: Loads, quantize, ballot, per-lane data
    // ------------------------------------------------------------------
    uint linear_block_32x32_index = gi >> 4;
    uint local_block_index = gi & 15u;  // 0..15 within 32x32 block
    uint local_x = local_block_index & 3u;
    uint local_y = local_block_index >> 2;

    // block_packing.comp: block32 = 2 * gl_WorkGroupID + (bit4, bit5 of the lane).
    int block32x32_x = 2 * int(grp.x) + int((gi >> 4) & 1u);
    int block32x32_y = 2 * int(grp.y) + int((gi >> 5) & 1u);

    // 8x8 block index
    int block8x8_x = 4 * block32x32_x + int(local_x);
    int block8x8_y = 4 * block32x32_y + int(local_y);

    bool in_range_8x8 = (block8x8_x < resolution_8x8_blocks_x) && (block8x8_y < resolution_8x8_blocks_y);
    bool in_range_32x32 = (block32x32_x < resolution_32x32_blocks_x) && (block32x32_y < resolution_32x32_blocks_y);

    // Load quant
    int quant = 0;
    if (in_range_32x32) {
        int bo32 = block_offset_32x32 + block_stride_32x32 * block32x32_y + block32x32_x;
        quant = int(t2.Load(bo32 * 4));
    }

    // Load the code word only. The payload offset is loaded once below,
    // used before the next barrier, and republished through groupshared.
    uint code_word = 0u;

    if (in_range_8x8) {
        int b8 = block_offset_8x8 + block_stride_8x8 * block8x8_y + block8x8_x;
        code_word = t0.Load(b8 * 8 + 0);
    }

    // Quantize code word
    code_word = quantize_code_word(code_word, quant);
    bool active = (code_word & 0xffffu) != 0u;

    // Ballot: InterlockedOr on groupshared
    uint bit_pos = local_y * 4u + local_x;
    if (active) {
        InterlockedOr(gs_ballot[blk], 1u << local_block_index);
    }

    // 64 sign bits is one per coefficient. The real signs are packed tightly
    // after the copy; this only reserves room. Do not load the payload offset
    // here. A second Load of that address is folded into a value that does
    // not survive the barrier, and every 8x8 then copies the first payload.
    uint plane_bytes = compute_required_8x8_size(code_word);
    uint bits_meta = active ? (plane_bytes * 8u + 64u + 24u) : 0u;
    gs_plane_bytes[gi] = plane_bytes;
    gs_sign_bits[gi] = 0u;
    gs_bits_meta[gi] = bits_meta;

    GroupMemoryBarrierWithGroupSync();  // B2

    // ------------------------------------------------------------------
    // P2: Sums and scans from groupshared; lane 15 allocates base
    // ------------------------------------------------------------------
    // Sum bits_meta over the 16 lanes of this block
    uint block_bits_meta = 0u;
    for (uint j = 0u; j < 16u; j++) {
        block_bits_meta += gs_bits_meta[blk * 16u + j];
    }

    uint total_words = (block_bits_meta + 31u) / 32u;
    if (total_words != 0u) total_words += 2u;  // HeaderSize = 2
    // Index order is coarse to fine. Packing the whole quant-0 frame is about
    // 1.3 MB and the caller cap holds ~180 KB. Keeping a fixed prefix makes the
    // packed byte count stable and skips the fine-band byte copy.
    {
        int packed_index = block_offset_32x32 + block_stride_32x32 * block32x32_y + block32x32_x;
        if (!in_range_32x32 || packed_index >= 96) total_words = 0u;
    }

    // Lane 15 of each block allocates
    bool writes_header = in_range_32x32 && (local_block_index == 15u);

    uint base = 0u;
    if (writes_header && total_words != 0u) {
        u2.InterlockedAdd(4, total_words, base);
    }

    // Only lane 15 publishes. Every lane writing here raced the real base back to 0.
    if (local_block_index == 15u) gs_base[blk] = base;
    GroupMemoryBarrierWithGroupSync();  // B3

    base = gs_base[blk];

    // Exclusive scan of plane bytes over the 16 lanes.
    uint exclusive_plane = 0u;
    for (uint jj = 0u; jj < local_block_index; jj++) {
        exclusive_plane += gs_plane_bytes[blk * 16u + jj];
    }

    // Total plane bytes for this block
    uint total_plane_bytes = 0u;
    for (uint mm = 0u; mm < 16u; mm++) {
        total_plane_bytes += gs_plane_bytes[blk * 16u + mm];
    }

    // Total active blocks (popcount of ballot)
    uint N = countbits(gs_ballot[blk]);

    // ------------------------------------------------------------------
    // P3: Header words, code word, q byte, planes and signs into gs_pkt
    // ------------------------------------------------------------------
    // Lane 15 writes header (words 0 and 1). An in-range block with no words
    // still publishes num_words 0 so a previous frame cannot leak through.
    int block_index = block_offset_32x32 + block_stride_32x32 * block32x32_y + block32x32_x;
    if (writes_header && total_words == 0u) {
        uint meta_offset_addr = uint(block_index) * 8u;
        u1.Store(int(meta_offset_addr), 0);
        u1.Store(int(meta_offset_addr + 4), 0);
    }
    if (writes_header && total_words != 0u) {
        // Word 0: ballot (16) | payload_words (12) << 16 | sequence (3) << 28 | extended(=0) << 31
        uint ballot_val = gs_ballot[blk] & 0xffffu;
        gs_pkt[blk * MAX_PACKET_WORDS + 0] = ballot_val | (total_words << 16) | (sequence_code << 28);

        // Word 1: modify_quant_code(quant_resolution_code, quant) (8) | block_index (24) << 8
        gs_pkt[blk * MAX_PACKET_WORDS + 1] = modify_quant_code(quant_resolution_code, quant) | (uint(block_index) << 8);

        // A block we cannot stage must not be advertised. A zero hole desyncs the decoder.
        if (total_words <= uint(MAX_PACKET_WORDS)) {
            uint meta_offset_addr = uint(block_index) * 8u;
            u1.Store(int(meta_offset_addr), int(base));
            u1.Store(int(meta_offset_addr + 4), int(total_words));
        }
    }

    // Code words, quant bytes, and magnitude planes are packet-local.
    // 4*base is the global address; adding it here wrote later blocks off the end of gs_pkt.
    uint local_sign_bits = 0u;
    if (active && total_words != 0u && total_words <= uint(MAX_PACKET_WORDS)) {
        uint hdr_offset = countbits(gs_ballot[blk] & ((1u << local_block_index) - 1u));

        uint cw_byte = 8u + 2u * hdr_offset;
        pkt_put_byte(blk, cw_byte, code_word & 0xffu);
        pkt_put_byte(blk, cw_byte + 1u, (code_word >> 8u) & 0xffu);

        uint q_byte_addr = 8u + 2u * N + hdr_offset;
        pkt_put_byte(blk, q_byte_addr, (code_word >> 16u) & 0xffu);

        // Only load of the payload offset. It stays in this block, before the
        // next barrier, and is not loaded anywhere earlier in the shader.
        uint meta_addr = uint(block_offset_8x8 + block_stride_8x8 * block8x8_y + block8x8_x) * 8u;
        uint meta_cw = t0.Load(int(meta_addr));
        uint in_q_bits = (meta_cw >> Q_PLANES_OFFSET) & 0xfu;
        uint out_q_bits = (code_word >> Q_PLANES_OFFSET) & 0xfu;
        uint in_offset = t0.Load(int(meta_addr + 4u));
        uint out_offset = 8u + 3u * N + exclusive_plane;

        for (int sb = 0; sb < 16; sb += 2) {
            uint out_planes = ((code_word >> uint(sb)) & 3u) + out_q_bits;
            uint in_planes = ((meta_cw >> uint(sb)) & 3u) + in_q_bits;
            if (in_planes != 0u) in_planes++;

            if (in_planes != 0u) {
                uint sign_plane = load_payload_byte(u2, in_offset);
                uint significant_mask = 0u;
                uint src = in_offset + 1u;
                uint dstb = out_offset;
                uint left = out_planes;
                while (left != 0u) {
                    uint src_addr = src + 8u;
                    uint src_word = u2.Load(src_addr & ~3u);
                    uint src_off = src_addr & 3u;
                    uint n = min(left, 4u - src_off);
                    uint chunk = (src_word >> (src_off * 8u)) & pkt_low_mask(n);
                    significant_mask |= chunk & 0xffu;
                    if (n > 1u) significant_mask |= (chunk >> 8u) & 0xffu;
                    if (n > 2u) significant_mask |= (chunk >> 16u) & 0xffu;
                    if (n > 3u) significant_mask |= chunk >> 24u;
                    uint dst_off = dstb & 3u;
                    uint n0 = min(n, 4u - dst_off);
                    pkt_or_bytes(blk, dstb, chunk, n0);
                    if (n0 < n) pkt_or_bytes(blk, dstb + n0, chunk >> (n0 * 8u), n - n0);
                    src += n;
                    dstb += n;
                    left -= n;
                }
                out_offset = dstb;

                uint sig_mask = significant_mask;
                while (sig_mask != 0u) {
                    uint coef = uint(firstbitlow(sig_mask));
                    sig_mask &= sig_mask - 1u;
                    uint sign_bit = (sign_plane >> coef) & 1u;
                    uint sword = (local_block_index << 1) + (local_sign_bits >> 5u);
                    uint sshift = local_sign_bits & 31u;
                    if (sword < 32u) InterlockedOr(gs_sign[blk][sword], sign_bit << sshift);
                    local_sign_bits++;
                }
            }

            in_offset += in_planes;
        }
    }
    gs_sign_bits[gi] = local_sign_bits;

    GroupMemoryBarrierWithGroupSync();  // B4

    // Tight sign bits, lane order, immediately after the magnitude planes.
    // Extra packet words past this are the unused per-coefficient reserve.
    if (in_range_32x32 && total_words != 0u && total_words <= uint(MAX_PACKET_WORDS) && local_block_index == 0u) {
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

    GroupMemoryBarrierWithGroupSync();

    if (in_range_32x32 && total_words != 0u && total_words <= uint(MAX_PACKET_WORDS)) {
        for (uint w = local_block_index; w < total_words; w += 16u) {
            u0.Store(int(4 * (base + w)), gs_pkt[blk * MAX_PACKET_WORDS + w]);
        }
    }
}
