/* OpenGL 4.3 port of src/shaders/dwt.hlsl from pyrowave-dx11, itself a port of
 * shaders/dwt.comp (Themaister, a0d2e71, FP16 off). The tile lives in an SSBO
 * so the four phases never read a value another group just wrote.
 */
static const char *kDwtComp = R"GL(#version 430
layout(local_size_x = 64) in;

layout(std140, binding = 0) uniform Push {
    ivec2 resolution;
    vec2 inv_resolution;
    ivec2 aligned_resolution;
    uint dc_shift;
    float range_scale;
    uint phase;
    uint groups_x;
    uint slot_stride;
    uint coeff_pitch;
    uint src_pitch;
    uint src_float;
    uint _pad0;
    uint _pad1;
};

layout(std430, binding = 1) readonly buffer SrcBytes { uint uSrc[]; };
layout(std430, binding = 2) buffer Scratch { float uScratch[]; };
layout(std430, binding = 3) buffer Coeff { float uCoeff[]; };
layout(std430, binding = 4) readonly buffer SrcFloat { float uSrcF[]; };

const float ALPHA = -1.586134342059924;
const float BETA = -0.052980118572961;
const float GAMMA = 0.882911075530934;
const float DELTA = 0.443506852043971;
const float K = 1.230174104914001;
const float inv_K = 1.0 / 1.230174104914001;

const int APRON = 4;
const int BLOCK_SIZE = 32;
const int BLOCK_SIZE_HALF = 16;
const int SHARED_STRIDE = 64;

uint local_index;
int read_slot;
int write_slot;

/* One tile is 20*64 vec2. Phase 1 writes the second slot; 20*64*2*8 = 20480 bytes,
 * under the 32 KB GL 4.3 shared-memory floor. Each group owns its tile. */
shared vec2 gs_tile[2][20 * 64];

vec2 load_shared(int y, int x)
{
    return gs_tile[read_slot][y * SHARED_STRIDE + x];
}

void store_shared(int y, int x, vec2 v)
{
    gs_tile[write_slot][y * SHARED_STRIDE + x] = v;
}

ivec2 unswizzle8x8(uint index)
{
    uint y = index & 1u;
    uint x = (index >> 1) & 3u;
    y |= ((index >> 3) & 3u) << 1;
    x |= ((index >> 5) & 1u) << 2;
    return ivec2(x, y);
}

int mirror_coord(int i, int R)
{
    int p = R * 2;
    int t = i % p;
    if (t < 0) t += p;
    return (t >= R) ? (p - 1 - t) : t;
}

int map_coord(int c, int R, int A)
{
    int g = c & ~1;
    if (g < 0) g -= 1;
    g += 1;
    int endc = 2 * A - R;
    int past = g + 2 * (R - A) + 1;
    g = (g >= endc) ? past : min(g, R);
    return ((c & 1) != 0) ? mirror_coord(g, R) : mirror_coord(g - 1, R);
}

float load_texel(int x, int y)
{
    x = map_coord(x, resolution.x, aligned_resolution.x);
    y = map_coord(y, resolution.y, aligned_resolution.y);
    float v;
    if (src_float != 0u) {
        uint idx = uint(y) * src_pitch + uint(x);
        v = uSrcF[idx];
    } else {
        uint idx = uint(y) * src_pitch + uint(x);
        uint word = uSrc[idx >> 2];
        uint shift = (idx & 3u) * 8u;
        v = float((word >> shift) & 255u) * (1.0 / 255.0);
    }
    v *= range_scale;
    if (dc_shift != 0u) v -= 0.5;
    return v;
}

vec4 gather_texels(ivec2 coord)
{
    return vec4(
        load_texel(coord.x, coord.y),
        load_texel(coord.x + 1, coord.y),
        load_texel(coord.x, coord.y + 1),
        load_texel(coord.x + 1, coord.y + 1));
}

void load_image_with_apron(uvec2 group_id)
{
    ivec2 base_coord = ivec2(group_id) * ivec2(BLOCK_SIZE) - ivec2(APRON);
    ivec2 local_coord0 = 2 * unswizzle8x8(local_index);
    ivec2 coord0 = base_coord + local_coord0;

    vec4 texels0 = gather_texels(coord0);
    vec4 texels1 = gather_texels(coord0 + ivec2(16, 0));
    vec4 texels2 = gather_texels(coord0 + ivec2(0, 16));
    vec4 texels3 = gather_texels(coord0 + ivec2(16, 16));

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
        ivec2 lc = ivec2(BLOCK_SIZE + 2 * int(local_index % 4u), 2 * int(local_index / 4u));
        vec4 t = gather_texels(base_coord + lc);
        store_shared(lc.y >> 1, lc.x + 0, t.xz);
        store_shared(lc.y >> 1, lc.x + 1, t.yw);
    }
    {
        ivec2 lc = ivec2(2 * int(local_index % 16u), BLOCK_SIZE + 2 * int(local_index / 16u));
        vec4 t = gather_texels(base_coord + lc);
        store_shared(lc.y >> 1, lc.x + 0, t.xz);
        store_shared(lc.y >> 1, lc.x + 1, t.yw);
    }
    if (local_index < 16u)
    {
        ivec2 lc = ivec2(BLOCK_SIZE + 2 * int(local_index % 4u), BLOCK_SIZE + 2 * int(local_index / 4u));
        vec4 t = gather_texels(base_coord + lc);
        store_shared(lc.y >> 1, lc.x + 0, t.xz);
        store_shared(lc.y >> 1, lc.x + 1, t.yw);
    }
}

void forward_transform8x2()
{
    vec2 values[16];
    ivec2 lc = ivec2(8 * int(local_index % 4u), int(local_index / 4u));

    for (int i0 = 0; i0 < 16; i0++) values[i0] = load_shared(lc.y, lc.x + i0);
    for (int i1 = 1; i1 < 15; i1 += 2) values[i1] += ALPHA * (values[i1 - 1] + values[i1 + 1]);
    for (int i2 = 2; i2 < 14; i2 += 2) values[i2] += BETA * (values[i2 - 1] + values[i2 + 1]);
    for (int i3 = 3; i3 < 13; i3 += 2) values[i3] += GAMMA * (values[i3 - 1] + values[i3 + 1]);
    for (int i4 = 4; i4 < 12; i4 += 2) values[i4] += DELTA * (values[i4 - 1] + values[i4 + 1]);

    for (int i5 = 2; i5 < 6; i5++)
    {
        vec2 a = values[2 * i5 + 0] * inv_K;
        vec2 b = values[2 * i5 + 1] * K;
        int yc = (lc.x >> 1) + (i5 - 2);
        store_shared(yc, 2 * lc.y + 0, vec2(a.x, b.x));
        store_shared(yc, 2 * lc.y + 1, vec2(a.y, b.y));
    }
}

void forward_transform4x2(bool active_lane, int y_offset)
{
    vec2 values[12];
    ivec2 lc = ivec2(4 * int(local_index % 8u), int(local_index / 8u) + y_offset);

    if (active_lane)
    {
        for (int i0 = 0; i0 < 12; i0++) values[i0] = load_shared(lc.y, lc.x + i0);
        for (int i1 = 1; i1 < 11; i1 += 2) values[i1] += ALPHA * (values[i1 - 1] + values[i1 + 1]);
        for (int i2 = 2; i2 < 10; i2 += 2) values[i2] += BETA * (values[i2 - 1] + values[i2 + 1]);
        for (int i3 = 3; i3 < 9; i3 += 2) values[i3] += GAMMA * (values[i3 - 1] + values[i3 + 1]);
        for (int i4 = 4; i4 < 8; i4 += 2) values[i4] += DELTA * (values[i4 - 1] + values[i4 + 1]);
    }
    else
    {
        for (int z = 0; z < 12; z++) values[z] = vec2(0.0);
    }

    if (active_lane)
    {
        for (int i5 = 2; i5 < 4; i5++)
        {
            vec2 a = values[2 * i5 + 0] * inv_K;
            vec2 b = values[2 * i5 + 1] * K;
            int yc = (lc.x >> 1) + (i5 - 2);
            store_shared(yc, 2 * lc.y + 0, vec2(a.x, b.x));
            store_shared(yc, 2 * lc.y + 1, vec2(a.y, b.y));
        }
    }
}

void main()
{
    local_index = gl_LocalInvocationIndex;

    /* Four phases of one group, in order. A shared-memory barrier replaces the
     * host dispatch between phases; no other group reads this tile. */
    read_slot = 0;
    write_slot = 0;
    load_image_with_apron(gl_WorkGroupID.xy);
    memoryBarrierShared();
    barrier();

    read_slot = 0;
    write_slot = 1;
    forward_transform8x2();
    forward_transform4x2(local_index < 32u, BLOCK_SIZE_HALF);
    memoryBarrierShared();
    barrier();

    read_slot = 1;
    write_slot = 0;
    forward_transform8x2();
    memoryBarrierShared();
    barrier();

    read_slot = 0;
    write_slot = 0;
    int out_w = aligned_resolution.x >> 1;
    int out_h = aligned_resolution.y >> 1;
    ivec2 lc = unswizzle8x8(local_index);
    for (int y = lc.y; y < BLOCK_SIZE_HALF; y += 8)
    {
        for (int x = lc.x * 2; x < BLOCK_SIZE; x += 16)
        {
            vec2 v0 = load_shared(y, x + 0);
            vec2 v1 = load_shared(y, x + 1);
            ivec2 c = ivec2(gl_WorkGroupID.xy) * (BLOCK_SIZE / 2) + ivec2(x >> 1, y);
            if (uint(c.x) < uint(out_w) && uint(c.y) < uint(out_h)) {
                uint pitch_h = (uint(out_h) + 31u) & ~31u;
                uint base = uint(c.y) * coeff_pitch + uint(c.x);
                uint layer_stride = coeff_pitch * pitch_h;
                uCoeff[0u * layer_stride + base] = v0.x;
                uCoeff[2u * layer_stride + base] = v0.y;
                uCoeff[1u * layer_stride + base] = v1.x;
                uCoeff[3u * layer_stride + base] = v1.y;
            }
        }
    }
}
)GL";
