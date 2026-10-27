// cs_5_0 twin of shaders/rdo_dwt.comp. Feature level 11_0. Explicit mad,
// integer fp16 (fxc deletes an immediate f32tof16/f16tof32 pair), integer
// mirrored-repeat gather, fp16 texel before the DC shift.
cbuffer Push : register(b0)
{
    int u_res_w;
    int u_res_h;
    int u_aligned_w;
    int u_aligned_h;
    int u_dc;
    int u_dst_w;
    int u_pad0;
    int u_pad1;
};

RWBuffer<float> Src : register(u0);
RWBuffer<float> Dst : register(u1);

static const float ALPHA = -1.586134342059924;
static const float BETA = -0.052980118572961;
static const float GAMMA = 0.882911075530934;
static const float DELTA = 0.443506852043971;
static const float KCOEF = 1.230174104914001;
static const float INV_K = 1.0 / KCOEF;

// Tile scratch is a UAV, not groupshared. On the HD 6670 a groupshared store of
// a packed fp16 pair clears the even dword (the odd neighbor stays intact).
// UAV dword stores of the same bits round-trip. u_pad1 is the group count in x.
RWBuffer<uint> Scratch : register(u2);
static int tile_x;
static int tile_y;
static int lane_id;

// The HD 6670 integer multiply does not return a full 32-bit product.
// Shift-add matches a 32-bit multiply for a 16-bit factor, which covers
// every resolution this encoder dispatches.
uint mul_u(uint a, uint b)
{
    // Products in this kernel stay below 2^23, so the f32 mantissa is exact.
    // A native integer multiply on TeraScale 2 does not return a full product,
    // and a 16-step shift-add of the constants (5, 1920) returned 0.
    // precise keeps fxc from turning this back into an integer multiply.
    precise float product = float(a) * float(b);
    return (uint)product;
}

int mul_i(int a, int b)
{
    return (int)mul_u((uint)a, (uint)b);
}

uint cell_index(int y, int x)
{
    uint tile = mul_u((uint)tile_y, (uint)u_pad1) + (uint)tile_x;
    return mul_u(tile, 884u) + mul_u((uint)y, 41u) + (uint)x;
}

// Same steps as rdo_dwt::f32_to_f16. An immediate f32tof16/f16tof32 pair is
// deleted, which skips the round that has to happen before the DC subtract.
// The integer sequence below is miscompiled on TeraScale 2 for some registers
// (it returns only the sign bit). PYRO_AMD uses the hardware opcodes, which
// match f32_to_f16 on values that already live in a register.
uint f16_from_bits(uint bits)
{
    uint sign = (bits >> 16) & 0x8000u;
    int exp = (int)((bits >> 23) & 0xffu);
    uint mant = bits & 0x007fffffu;
    if (exp == 255) {
        if (mant == 0u) {
            return sign | 0x7c00u;
        }
        uint nan = mant >> 13;
        if (nan == 0u) {
            nan = 1u;
        }
        return sign | 0x7c00u | nan;
    }
    if (exp == 0) {
        return sign;
    }
    int e = exp - 127 + 15;
    if (e >= 31) {
        return sign | 0x7c00u;
    }
    if (e <= 0) {
        if (e < -10) {
            return sign;
        }
        uint m = mant | 0x00800000u;
        uint shift = (uint)(14 - e);
        uint hsub = m >> shift;
        uint rem = m & ((1u << shift) - 1u);
        uint tie = 1u << (shift - 1u);
        if (rem > tie || (rem == tie && (hsub & 1u) == 1u)) {
            hsub += 1u;
        }
        return sign | hsub;
    }
    uint h = mant >> 13;
    uint round = (mant >> 12) & 1u;
    uint sticky = mant & 0x0fffu;
    if (round == 1u && (sticky != 0u || (h & 1u) == 1u)) {
        h += 1u;
        if (h == 0x400u) {
            h = 0u;
            e += 1;
            if (e >= 31) {
                return sign | 0x7c00u;
            }
        }
    }
    return sign | ((uint)e << 10) | h;
}

uint f32_to_f16(float val)
{
    uint bits = asuint(val);
    uint result = 0u;
#ifdef PYRO_AMD
    // The integer converter returns only the sign bit for some live registers.
    // Hardware f32tof16 stays correct on those, except it truncates where
    // RTNE rounds up. Prefer the integer result, and fix the hardware result
    // when the integer path collapses. PYRO_DWT_PROBE 5 forces hardware and
    // 6 forces the integer converter.
    uint hw = f32tof16(val);
    bits ^= (uint)u_res_w;
    bits ^= (uint)u_res_w;
    uint soft = f16_from_bits(bits);
    uint mant = bits & 0x007fffffu;
    uint roundb = (mant >> 12) & 1u;
    uint sticky = mant & 0x0fffu;
    uint exp = (bits >> 23) & 0xffu;
    int e = (int)exp - 127 + 15;
    uint sign = hw & 0x8000u;
    uint fixed = hw;
    // Hardware sometimes truncates and sometimes already rounded. Add the
    // RTNE bump only when the opcode still holds the truncated encoding.
    if (exp > 0u && exp < 255u && e > 0 && e < 31) {
        uint h = mant >> 13;
        uint trunc = sign | ((uint)e << 10) | h;
        if (hw == trunc && roundb == 1u && (sticky != 0u || (h & 1u) == 1u)) {
            fixed = sign | ((hw & 0x7fffu) + 1u);
        }
    }
    if (u_pad0 == 6) {
        result = soft;
    } else if (u_pad0 == 7 || u_pad0 == 8) {
        result = hw;
    } else if (u_pad0 == 5 || u_pad0 == 9) {
        result = fixed;
    } else if ((soft & 0x7fffu) == 0u && (hw & 0x7fffu) != 0u) {
        result = fixed;
    } else {
        result = soft;
    }
#else
    result = f16_from_bits(bits);
#endif
    return result;
}

float f16_to_f32(uint half_bits)
{
#ifdef PYRO_AMD
    // 8 and 9 select the hardware expand. The integer expand is mis-scheduled
    // for some live values on TeraScale 2, the same way the integer pack is.
    if (u_pad0 == 8 || u_pad0 == 9) {
        return f16tof32(half_bits);
    }
#endif
    uint sign = (half_bits & 0x8000u) << 16;
    uint exp = (half_bits >> 10) & 0x1fu;
    uint mant = half_bits & 0x03ffu;
    uint bits;
    if (exp == 0u) {
        if (mant == 0u) {
            bits = sign;
        } else {
            uint m = mant;
            int e = 127 - 15 + 1;
            [loop]
            for (int s0 = 0; s0 < 10; s0++) {
                if ((m & 0x400u) != 0u) {
                    break;
                }
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ffu;
            bits = sign | ((uint)e << 23) | (m << 13);
        }
    } else if (exp == 31u) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (uint)(127 - 15)) << 23) | (mant << 13);
    }
    return asfloat(bits);
}

float round16(float x)
{
    return f16_to_f32(f32_to_f16(x));
}

int wrap_mirror(int i, int size)
{
    if (size <= 1) {
        return 0;
    }
    // One reflection. Every gather this kernel builds sits in (-2*size, 2*size),
    // so the fold is a subtract. A hardware udiv is wrong on TeraScale 2 even
    // when the quotient is 0, and lavapipe mis-lowers a non-power-of-two `%`.
    // Same result as rdo_dwt::wrap_mirror for that range (wrap_mirror(-1, 96) == 0).
    int n = size << 1;
    if (i < 0) {
        int q = -i;
        i = (q == 0) ? 0 : n - q;
    } else if (i >= n) {
        i -= n;
    }
    if (i >= size) {
        i = n - 1 - i;
    }
    return i;
}

int gather_raw(int coord, int resolution, int aligned)
{
    int c = coord;
    if (c < 0) {
        c -= 1;
    }
    c += 1;
    int end = 2 * aligned - resolution;
    int past = c + 2 * (resolution - aligned) + 1;
    int mapped = (c >= end) ? past : min(c, resolution);
    // floor(mapped * (1/res) * res - 0.5) == mapped - 1 for every resolution
    // in 2..4096 (rdo_dwt::gather_raw). The reciprocal is not exact on TeraScale 2.
    return mapped - 1;
}

float at_texel(int ix, int iy)
{
    uint xx = (uint)wrap_mirror(ix, u_res_w);
    uint yy = (uint)wrap_mirror(iy, u_res_h);
    uint index = mul_u(yy, (uint)u_res_w) + xx;
    float v = round16(Src[index]);
    if (u_dc != 0) {
        v -= 0.5;
    }
    return v;
}

float4 gather4(int x, int y)
{
    int i0 = gather_raw(x, u_res_w, u_aligned_w);
    int j0 = gather_raw(y, u_res_h, u_aligned_h);
    // Separate statements. One expression of four calls dropped a tap on
    // TeraScale 2 (it came back as -0.5 on a flat field).
    float a = at_texel(i0, j0);
    float b = at_texel(i0 + 1, j0);
    float c = at_texel(i0, j0 + 1);
    float d = at_texel(i0 + 1, j0 + 1);
    return float4(a, b, c, d);
}

void unswizzle(int index, out int x, out int y)
{
    y = (index & 1) | (((index >> 3) & 3) << 1);
    x = ((index >> 1) & 3) | (((index >> 5) & 1) << 2);
}

float2 ld(int y, int x)
{
    uint bits = Scratch[cell_index(y, x)];
    // Separate calls. One expression of several calls drops a result on
    // TeraScale 2 (gather4 did this; the high half of a pair is LH).
    float lo = f16_to_f32(bits & 0xffffu);
    float hi = f16_to_f32(bits >> 16);
    return float2(lo, hi);
}

void st(int y, int x, float2 v)
{
    uint lo = f32_to_f16(v.x);
    uint hi = f32_to_f16(v.y);
    Scratch[cell_index(y, x)] = lo | (hi << 16);
}

// TeraScale 2 has no fused multiply-add. `mad` is a rounded multiply plus a
// rounded add, and that is a different f16 code from f32::mul_add on about 6%
// of a detailed frame. Dekker's product plus a two-sum is bit-exact for the
// values this lift produces (checked against the CPU wavelet). `precise` keeps
// fxc from contracting the error terms back into a mad.
float soft_fma(float a, float b, float c)
{
    precise float ph = a * b;
    precise float ax = 4097.0f * a;
    precise float ahi = ax - (ax - a);
    precise float alo = a - ahi;
    precise float bx = 4097.0f * b;
    precise float bhi = bx - (bx - b);
    precise float blo = b - bhi;
    precise float t0 = ahi * bhi;
    precise float t1 = ph - t0;
    precise float t2 = alo * bhi;
    precise float t3 = t1 - t2;
    precise float t4 = ahi * blo;
    precise float t5 = t3 - t4;
    precise float t6 = alo * blo;
    precise float pl = t6 - t5;
    precise float sh = c + ph;
    precise float bb = sh - c;
    precise float sl = (c - (sh - bb)) + (ph - bb);
    precise float err = sl + pl;
    precise float result = sh + err;
    return result;
}

float madd(float a, float b, float c)
{
#ifdef PYRO_AMD
    return soft_fma(a, b, c);
#else
    return mad(a, b, c);
#endif
}

void lift(inout float2 values[16], int n)
{
    for (int a = 1; a < n - 1; a += 2) {
        float2 sum = values[a - 1] + values[a + 1];
        float ox = madd(ALPHA, sum.x, values[a].x);
        float oy = madd(ALPHA, sum.y, values[a].y);
        values[a] = float2(ox, oy);
    }
    for (int b = 2; b < n - 2; b += 2) {
        float2 sum = values[b - 1] + values[b + 1];
        float ox = madd(BETA, sum.x, values[b].x);
        float oy = madd(BETA, sum.y, values[b].y);
        values[b] = float2(ox, oy);
    }
    for (int g = 3; g < n - 3; g += 2) {
        float2 sum = values[g - 1] + values[g + 1];
        float ox = madd(GAMMA, sum.x, values[g].x);
        float oy = madd(GAMMA, sum.y, values[g].y);
        values[g] = float2(ox, oy);
    }
    for (int d = 4; d < n - 4; d += 2) {
        float2 sum = values[d - 1] + values[d + 1];
        float ox = madd(DELTA, sum.x, values[d].x);
        float oy = madd(DELTA, sum.y, values[d].y);
        values[d] = float2(ox, oy);
    }
}

void load_apron(int local_in, int gx, int gy)
{
    uint local = (uint)local_in;
    int base_x = gx * 32 - 4;
    int base_y = gy * 32 - 4;
    int ux;
    int uy;
    unswizzle(local_in, ux, uy);
    int lx = 2 * ux;
    int ly = 2 * uy;
    float4 t0 = gather4(base_x + lx, base_y + ly);
    float4 t1 = gather4(base_x + lx + 16, base_y + ly);
    float4 t2 = gather4(base_x + lx, base_y + ly + 16);
    float4 t3 = gather4(base_x + lx + 16, base_y + ly + 16);
    int sy = ly >> 1;
    st(sy, lx, float2(t0.x, t0.z));
    st(sy, lx + 1, float2(t0.y, t0.w));
    st(sy, lx + 16, float2(t1.x, t1.z));
    st(sy, lx + 17, float2(t1.y, t1.w));
    st(sy + 8, lx, float2(t2.x, t2.z));
    st(sy + 8, lx + 1, float2(t2.y, t2.w));
    st(sy + 8, lx + 16, float2(t3.x, t3.z));
    st(sy + 8, lx + 17, float2(t3.y, t3.w));

    int ax = int(32u + 2u * (local & 3u));
    int ay = int(2u * (local >> 2));
    float4 tr = gather4(base_x + ax, base_y + ay);
    st(ay >> 1, ax, float2(tr.x, tr.z));
    st(ay >> 1, ax + 1, float2(tr.y, tr.w));

    int bx = int(2u * (local & 15u));
    int by = int(32u + 2u * (local >> 4));
    float4 bl = gather4(base_x + bx, base_y + by);
    st(by >> 1, bx, float2(bl.x, bl.z));
    st(by >> 1, bx + 1, float2(bl.y, bl.w));

    if (local < 16u) {
        int cx = int(32u + 2u * (local & 3u));
        int cy = int(32u + 2u * (local >> 2));
        float4 br = gather4(base_x + cx, base_y + cy);
        st(cy >> 1, cx, float2(br.x, br.z));
        st(cy >> 1, cx + 1, float2(br.y, br.w));
    }
}

void xform8(int local_in)
{
    uint local = (uint)local_in;
    float2 values[16];
    float2 outs[8];
    int oy[8];
    int ox[8];
    int local_coord_x = int(8u * (local & 3u));
    int local_coord_y = int(local >> 2);
    [unroll]
    for (int i0 = 0; i0 < 16; i0++) {
        values[i0] = ld(local_coord_y, local_coord_x + i0);
    }
    lift(values, 16);
    int n = 0;
    [unroll]
    for (int i1 = 2; i1 < 6; i1++) {
        float2 a = values[2 * i1] * INV_K;
        float2 b = values[2 * i1 + 1] * KCOEF;
        int y = (local_coord_x >> 1) + (i1 - 2);
        int x = 2 * local_coord_y;
        oy[n] = y;
        ox[n] = x;
        outs[n] = float2(a.x, b.x);
        n++;
        oy[n] = y;
        ox[n] = x + 1;
        outs[n] = float2(a.y, b.y);
        n++;
    }
    AllMemoryBarrierWithGroupSync();
    [unroll]
    for (int k0 = 0; k0 < 8; k0++) {
        st(oy[k0], ox[k0], outs[k0]);
    }
    AllMemoryBarrierWithGroupSync();
}

void xform4(int local_in)
{
    uint local = (uint)local_in;
    float2 values[16];
    float2 outs[4];
    int oy[4];
    int ox[4];
    int n = 0;
    [unroll]
    for (int z0 = 0; z0 < 4; z0++) {
        outs[z0] = 0;
        oy[z0] = 0;
        ox[z0] = 0;
    }
    if (local < 32u) {
        int local_coord_x = int(4u * (local & 7u));
        int local_coord_y = int(local >> 3) + 16;
        [unroll]
        for (int i2 = 0; i2 < 12; i2++) {
            values[i2] = ld(local_coord_y, local_coord_x + i2);
        }
        lift(values, 12);
        [unroll]
        for (int i3 = 2; i3 < 4; i3++) {
            float2 a = values[2 * i3] * INV_K;
            float2 b = values[2 * i3 + 1] * KCOEF;
            int y = (local_coord_x >> 1) + (i3 - 2);
            int x = 2 * local_coord_y;
            oy[n] = y;
            ox[n] = x;
            outs[n] = float2(a.x, b.x);
            n++;
            oy[n] = y;
            ox[n] = x + 1;
            outs[n] = float2(a.y, b.y);
            n++;
        }
    }
    AllMemoryBarrierWithGroupSync();
    [unroll]
    for (int k1 = 0; k1 < 4; k1++) {
        if (k1 < n) {
            st(oy[k1], ox[k1], outs[k1]);
        }
    }
    AllMemoryBarrierWithGroupSync();
}

void store_out(int local, int gx, int gy)
{
    int ow = u_dst_w;
    int oh = int((uint)u_aligned_h >> 1);
    int lx;
    int ly0;
    unswizzle(local, lx, ly0);
    for (int y0 = ly0; y0 < 16; y0 += 8) {
        for (int x0 = lx * 2; x0 < 32; x0 += 16) {
            float2 v0 = ld(y0, x0);
            float2 v1 = ld(y0, x0 + 1);
            int img_x = gx * 16 + (x0 >> 1);
            int img_y = gy * 16 + y0;
            if (img_x >= 0 && img_y >= 0 && img_x < ow && img_y < oh) {
                int at = mul_i(img_y, ow) + img_x;
                int span = mul_i(ow, oh);
                Dst[at] = v0.x;
                Dst[span + span + at] = v0.y;
                Dst[span + at] = v1.x;
                Dst[span + span + span + at] = v1.y;
            }
        }
    }
}

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID, uint lid : SV_GroupIndex)
{
    int local = (int)lid;
    int gx = (int)gid.x;
    int gy = (int)gid.y;
    tile_x = gx;
    tile_y = gy;
    lane_id = local;
    uint tile = mul_u((uint)gy, (uint)u_pad1) + (uint)gx;
    uint base = mul_u(tile, 884u);
    [loop]
    for (int z1 = local; z1 < 820; z1 += 64) {
        Scratch[base + (uint)z1] = 0u;
    }
    AllMemoryBarrierWithGroupSync();
    load_apron(local, gx, gy);
    AllMemoryBarrierWithGroupSync();
    xform8(local);
    xform4(local);
    xform8(local);
    store_out(local, gx, gy);
}
