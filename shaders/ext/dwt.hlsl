// D3D11 (cs_5_0) port of PyroWave shaders/dwt.comp @ a0d2e71 (PRECISION=1).
// One level per dispatch, addressed by SV_GroupID.xy. The apron lives in
// groupshared and the lifting phases use the same barriers as dwt.comp.
// Inputs are texture gathers. Texture2D.Load past texel 7 is wrong on the
// HD 6670. Record 0 of the 192-byte cbuffer is this level. store_mask bits
// 0..3 select coefficient layers. Bit 4 writes LL to the texture the next
// level gathers and does not write that sample to the coefficient buffer.
cbuffer Push : register(b0)
{
    uint4 jobw[12];
};

static int2 resolution;
static float2 inv_resolution;
static int2 aligned_resolution;
static uint dc_shift;
static float range_scale;
static uint coeff_pitch;
static uint store_mask;

// t0-t2 stay free for the coefficient buffers the other stages bind.
// SV_GroupID.z picks the plane. A level's Y, Cb and Cr share one grid.
// SV_GroupID.z picks one component. The branch is uniform, so the other
// planes are not fetched.
Texture2DArray<float> uPlane0 : register(t3);
Texture2DArray<float> uPlane1 : register(t4);
Texture2DArray<float> uPlane2 : register(t5);
SamplerState uSamp : register(s0);
RWTexture2DArray<float> uOutput : register(u0);
RWBuffer<float> uCoeff : register(u1);
RWTexture2DArray<float> uOutput1 : register(u2);
RWBuffer<float> uCoeff1 : register(u3);
RWTexture2DArray<float> uOutput2 : register(u4);
RWBuffer<float> uCoeff2 : register(u5);

#define APRON 4
#define APRON_HALF 2
#define BLOCK_SIZE 32
#define BLOCK_SIZE_HALF 16

static const float ALPHA = -1.586134342059924;
static const float BETA = -0.052980118572961;
static const float GAMMA = 0.882911075530934;
static const float DELTA = 0.443506852043971;
static const float K = 1.230174104914001;
static const float inv_K = 1.0 / 1.230174104914001;

#define SHARED_ROWS ((BLOCK_SIZE + 2 * APRON) / 2)
// +1 matches dwt_common.h so consecutive rows do not land on the same banks.
#define SHARED_STRIDE ((BLOCK_SIZE + 2 * APRON) + 1)
// PRECISION=1: fp32 lifting, fp16 at every shared store (dwt_common.h).
groupshared uint gs_tile[SHARED_ROWS][SHARED_STRIDE];
static uint local_index;

float2 load_shared(int y, int x)
{
    uint v = gs_tile[y][x];
    return float2(f16tof32(v & 0xffffu), f16tof32(v >> 16));
}

void store_shared(int y, int x, float2 v)
{
    gs_tile[y][x] = f32tof16(v.x) | (f32tof16(v.y) << 16);
}

int2 unswizzle8x8(uint index)
{
    uint y = index & 1u;
    uint x = (index >> 1) & 3u;
    y |= ((index >> 3) & 3u) << 1;
    x |= ((index >> 5) & 1u) << 2;
    return int2(x, y);
}

// dwt.comp:30-34. A negative component is decremented and then incremented,
// so it is unchanged. A non-negative component is incremented.
float2 generate_mirror_uv(int2 coord)
{
    coord -= int2(coord < int2(0, 0));
    coord += 1;
    int2 end_mirrored_clamp = 2 * aligned_resolution - resolution;
    int2 past_wrapped_coord = coord + 2 * (resolution - aligned_resolution) + 1;
    coord = (coord >= end_mirrored_clamp) ? past_wrapped_coord : min(coord, resolution);
    return float2(coord) * inv_resolution;
}

// .wzxy is the 2x2 at (x,y), (x+1,y), (x,y+1), (x+1,y+1). Slice 0.
float4 gather_plane(int2 coord, uint slot)
{
    float3 uvw = float3(generate_mirror_uv(coord), 0.0);
    float4 t;
    [branch] if (slot == 1u) t = uPlane1.Gather(uSamp, uvw).wzxy;
    else [branch] if (slot == 2u) t = uPlane2.Gather(uSamp, uvw).wzxy;
    else t = uPlane0.Gather(uSamp, uvw).wzxy;
    t *= range_scale;
    if (dc_shift != 0u) t -= 0.5;
    return t;
}

void store_ll(uint slot, int2 c, float v)
{
    [branch] if (slot == 1u) uOutput1[int3(c, 0)] = v;
    else [branch] if (slot == 2u) uOutput2[int3(c, 0)] = v;
    else uOutput[int3(c, 0)] = v;
}

void store_coeff(uint slot, uint elem, float v)
{
    [branch] if (slot == 1u) uCoeff1[elem] = v;
    else [branch] if (slot == 2u) uCoeff2[elem] = v;
    else uCoeff[elem] = v;
}

void load_image_with_apron(uint2 group_id, uint slot)
{
    int2 base_coord = int2(group_id) * int2(BLOCK_SIZE, BLOCK_SIZE) - APRON;
    int2 local_coord0 = 2 * unswizzle8x8(local_index);
    int2 coord0 = base_coord + local_coord0;

    float4 texels0 = gather_plane(coord0, slot);
    float4 texels1 = gather_plane(coord0 + int2(16, 0), slot);
    float4 texels2 = gather_plane(coord0 + int2(0, 16), slot);
    float4 texels3 = gather_plane(coord0 + int2(16, 16), slot);

    int ly = local_coord0.y >> 1;
    store_shared(ly + 0, local_coord0.x + 0, texels0.xz);
    store_shared(ly + 0, local_coord0.x + 1, texels0.yw);
    store_shared(ly + 0, local_coord0.x + 16, texels1.xz);
    store_shared(ly + 0, local_coord0.x + 17, texels1.yw);
    store_shared(ly + 8, local_coord0.x + 0, texels2.xz);
    store_shared(ly + 8, local_coord0.x + 1, texels2.yw);
    store_shared(ly + 8, local_coord0.x + 16, texels3.xz);
    store_shared(ly + 8, local_coord0.x + 17, texels3.yw);

    {
        int2 lc = int2(BLOCK_SIZE + 2 * (local_index % 4u), 2 * (local_index / 4u));
        float4 t = gather_plane(base_coord + lc, slot);
        store_shared(lc.y >> 1, lc.x + 0, t.xz);
        store_shared(lc.y >> 1, lc.x + 1, t.yw);
    }
    {
        int2 lc = int2(2 * (local_index % 16u), BLOCK_SIZE + 2 * (local_index / 16u));
        float4 t = gather_plane(base_coord + lc, slot);
        store_shared(lc.y >> 1, lc.x + 0, t.xz);
        store_shared(lc.y >> 1, lc.x + 1, t.yw);
    }
    if (local_index < 16u)
    {
        int2 lc = int2(BLOCK_SIZE + 2 * (local_index % 4u), BLOCK_SIZE + 2 * (local_index / 4u));
        float4 t = gather_plane(base_coord + lc, slot);
        store_shared(lc.y >> 1, lc.x + 0, t.xz);
        store_shared(lc.y >> 1, lc.x + 1, t.yw);
    }
}

void forward_transform8x2()
{
    float2 values[16];
    int2 lc = int2(8 * (local_index % 4u), local_index / 4u);

    [unroll] for (int i0 = 0; i0 < 16; i0++) values[i0] = load_shared(lc.y, lc.x + i0);
    [unroll] for (int i1 = 1; i1 < 15; i1 += 2) values[i1] += ALPHA * (values[i1 - 1] + values[i1 + 1]);
    [unroll] for (int i2 = 2; i2 < 14; i2 += 2) values[i2] += BETA * (values[i2 - 1] + values[i2 + 1]);
    [unroll] for (int i3 = 3; i3 < 13; i3 += 2) values[i3] += GAMMA * (values[i3 - 1] + values[i3 + 1]);
    [unroll] for (int i4 = 4; i4 < 12; i4 += 2) values[i4] += DELTA * (values[i4 - 1] + values[i4 + 1]);

    // dwt.comp:124. Loads are in registers; this publishes them before the stores.
    GroupMemoryBarrierWithGroupSync();

    [unroll] for (int i5 = 2; i5 < 6; i5++)
    {
        float2 a = values[2 * i5 + 0] * inv_K;
        float2 b = values[2 * i5 + 1] * K;
        int yc = (lc.x >> 1) + (i5 - 2);
        store_shared(yc, 2 * lc.y + 0, float2(a.x, b.x));
        store_shared(yc, 2 * lc.y + 1, float2(a.y, b.y));
    }
}

void forward_transform4x2(bool active_lane, int y_offset)
{
    float2 values[12];
    int2 lc = int2(4 * (local_index % 8u), local_index / 8u + y_offset);

    if (active_lane)
    {
        [unroll] for (int i0 = 0; i0 < 12; i0++) values[i0] = load_shared(lc.y, lc.x + i0);
        [unroll] for (int i1 = 1; i1 < 11; i1 += 2) values[i1] += ALPHA * (values[i1 - 1] + values[i1 + 1]);
        [unroll] for (int i2 = 2; i2 < 10; i2 += 2) values[i2] += BETA * (values[i2 - 1] + values[i2 + 1]);
        [unroll] for (int i3 = 3; i3 < 9; i3 += 2) values[i3] += GAMMA * (values[i3 - 1] + values[i3 + 1]);
        [unroll] for (int i4 = 4; i4 < 8; i4 += 2) values[i4] += DELTA * (values[i4 - 1] + values[i4 + 1]);
    }
    else
    {
        [unroll] for (int z = 0; z < 12; z++) values[z] = float2(0.0, 0.0);
    }

    // dwt.comp:176. Uniform control flow, outside the active_lane branch.
    GroupMemoryBarrierWithGroupSync();

    if (active_lane)
    {
        [unroll] for (int i5 = 2; i5 < 4; i5++)
        {
            float2 a = values[2 * i5 + 0] * inv_K;
            float2 b = values[2 * i5 + 1] * K;
            int yc = (lc.x >> 1) + (i5 - 2);
            store_shared(yc, 2 * lc.y + 0, float2(a.x, b.x));
            store_shared(yc, 2 * lc.y + 1, float2(a.y, b.y));
        }
    }
}

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID, uint gi : SV_GroupIndex)
{
    local_index = gi;
    uint2 tile = gid.xy;
    uint z = gid.z;
    if (z > 2u) z = 0u;
    uint4 a, b, c, d;
    [branch] if (z == 1u) {
        a = jobw[4]; b = jobw[5]; c = jobw[6]; d = jobw[7];
    } else [branch] if (z == 2u) {
        a = jobw[8]; b = jobw[9]; c = jobw[10]; d = jobw[11];
    } else {
        a = jobw[0]; b = jobw[1]; c = jobw[2]; d = jobw[3];
    }
    resolution = int2(asint(a.x), asint(a.y));
    // a.z holds the linear group begin on the old fused path. It is not inv_resolution.
    inv_resolution = float2(1.0 / float(resolution.x), 1.0 / float(resolution.y));
    aligned_resolution = int2(asint(b.x), asint(b.y));
    dc_shift = b.z;
    range_scale = asfloat(b.w);
    coeff_pitch = c.w;
    store_mask = d.z;

    load_image_with_apron(tile, z);
    GroupMemoryBarrierWithGroupSync(); // dwt.comp:207
    forward_transform8x2();
    forward_transform4x2(local_index < 32u, BLOCK_SIZE_HALF);
    GroupMemoryBarrierWithGroupSync(); // dwt.comp:215
    forward_transform8x2();
    GroupMemoryBarrierWithGroupSync(); // dwt.comp:220

    int out_w = aligned_resolution.x >> 1;
    int out_h = aligned_resolution.y >> 1;
    uint pitch_h = (uint(out_h) + 31u) & ~31u;
    uint layer_stride = coeff_pitch * pitch_h;
    uint keep = store_mask;
    // store_mask 0 is the test sentinel: every coefficient layer and the LL
    // texture. A live level never sends 0.
    if (keep == 0u) keep = 0x1fu;
    int2 lc = unswizzle8x8(local_index);
    for (int y = lc.y; y < BLOCK_SIZE_HALF; y += 8)
    {
        for (int x = lc.x * 2; x < BLOCK_SIZE; x += 16)
        {
            float2 v0 = load_shared(y, x + 0);
            float2 v1 = load_shared(y, x + 1);
            int2 oc = int2(tile) * (BLOCK_SIZE / 2) + int2(x >> 1, y);
            if ((uint)oc.x < (uint)out_w && (uint)oc.y < (uint)out_h) {
                if ((keep & 16u) != 0u) store_ll(z, oc, v0.x);
                if ((keep & 15u) != 0u) {
                    uint base = uint(oc.y) * coeff_pitch + uint(oc.x);
                    if ((keep & 1u) != 0u) store_coeff(z, base, v0.x);
                    if ((keep & 4u) != 0u) store_coeff(z, 2u * layer_stride + base, v0.y);
                    if ((keep & 2u) != 0u) store_coeff(z, 1u * layer_stride + base, v1.x);
                    if ((keep & 8u) != 0u) store_coeff(z, 3u * layer_stride + base, v1.y);
                }
            }
        }
    }
}
