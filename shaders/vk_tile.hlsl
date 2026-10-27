// cs_5_0 tile wavelet. One group per 32x32 luma tile: CDF 9/7, quantize, pack.
cbuffer Push : register(b0) {
    uint width;
    uint height;
    uint cw;
    uint ch;
    uint prefix;
    uint energy;
    uint2 pad;
};

ByteAddressBuffer Src : register(t0);
RWByteAddressBuffer Dst : register(u0);

static const float ALPHA = -1.586134342059924;
static const float BETA = -0.052980118572961;
static const float GAMMA = 0.882911075530934;
static const float DELTA = 0.443506852043971;
static const float KSCALE = 1.230174104914001;
static const float INV_K = 1.0 / KSCALE;

groupshared float s[1024];
groupshared float red[32];

float ld(uint base, uint stride, uint i) {
    return s[base + i * stride];
}

void st(uint base, uint stride, uint i, float v) {
    s[base + i * stride] = v;
}

void lift(uint base, uint stride, uint n) {
    uint i = 0u;
    if (n < 2u) {
        return;
    }
    for (i = 1u; i + 1u < n; i += 2u) {
        st(base, stride, i, ld(base, stride, i) + ALPHA * (ld(base, stride, i - 1u) + ld(base, stride, i + 1u)));
    }
    if ((n & 1u) == 0u) {
        i = n - 1u;
        st(base, stride, i, ld(base, stride, i) + ALPHA * (ld(base, stride, i - 1u) + ld(base, stride, i - 1u)));
    }
    for (i = 2u; i + 1u < n; i += 2u) {
        st(base, stride, i, ld(base, stride, i) + BETA * (ld(base, stride, i - 1u) + ld(base, stride, i + 1u)));
    }
    if ((n & 1u) == 1u && n >= 3u) {
        i = n - 1u;
        st(base, stride, i, ld(base, stride, i) + BETA * (ld(base, stride, i - 1u) + ld(base, stride, i - 1u)));
    }
    for (i = 3u; i + 1u < n; i += 2u) {
        st(base, stride, i, ld(base, stride, i) + GAMMA * (ld(base, stride, i - 1u) + ld(base, stride, i + 1u)));
    }
    if ((n & 1u) == 0u && n >= 4u) {
        i = n - 1u;
        st(base, stride, i, ld(base, stride, i) + GAMMA * (ld(base, stride, i - 1u) + ld(base, stride, i - 1u)));
    }
    for (i = 4u; i + 1u < n; i += 2u) {
        st(base, stride, i, ld(base, stride, i) + DELTA * (ld(base, stride, i - 1u) + ld(base, stride, i + 1u)));
    }
    if (n >= 5u && (n & 1u) == 1u) {
        i = n - 1u;
        st(base, stride, i, ld(base, stride, i) + DELTA * (ld(base, stride, i - 1u) + ld(base, stride, i - 1u)));
    }
    for (i = 0u; i < n; i += 2u) {
        st(base, stride, i, ld(base, stride, i) * INV_K);
        if (i + 1u < n) {
            st(base, stride, i + 1u, ld(base, stride, i + 1u) * KSCALE);
        }
    }
}

uint byte_at(uint off) {
    uint word = Src.Load(off & ~3u);
    return (word >> ((off & 3u) * 8u)) & 0xffu;
}

[numthreads(32, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid3 : SV_GroupThreadID) {
    uint ltx = (width + 31u) >> 5;
    uint id = gid.x;
    // Keep every cbuffer member live under /WX without changing the quant.
    float cbuf_bias = float(cw ^ ch ^ pad.x ^ pad.y) * 0.0;
    uint pw = width;
    uint ph = height;
    uint ox = (id % ltx) << 5;
    uint oy = (id / ltx) << 5;
    uint bw = min(32u, pw - ox);
    uint bh = min(32u, ph - oy);
    uint tid = tid3.x;

    for (uint i = tid; i < 1024u; i += 32u) {
        s[i] = 0.0;
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint y = tid; y < bh; y += 32u) {
        uint row = (oy + y) * pw + ox;
        for (uint x = 0u; x < bw; x += 1u) {
            s[y * 32u + x] = (float)byte_at(row + x) - 128.0;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid < bh) {
        lift(tid * 32u, 1u, bw);
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid < bw) {
        lift(tid, 32u, bh);
    }
    GroupMemoryBarrierWithGroupSync();

    float m = 0.0;
    for (uint k = tid; k < 1024u; k += 32u) {
        uint col = k & 31u;
        uint yy = k >> 5;
        float q = 0.0;
        if (col < bw && yy < bh) {
            q = clamp(round((s[k] + cbuf_bias) * 4.0), -32767.0, 32767.0);
            m = max(m, abs(q));
        }
        s[k] = q;
    }
    red[tid] = m;
    GroupMemoryBarrierWithGroupSync();
    if (id < prefix) {
        for (uint w = tid; w < 512u; w += 32u) {
            uint e0 = w * 2u;
            uint e1 = e0 + 1u;
            uint a = ((uint)(int)s[e0]) & 0xffffu;
            uint b = ((uint)(int)s[e1]) & 0xffffu;
            Dst.Store(id * 2048u + w * 4u, a | (b << 16));
        }
    }
    if (tid == 0u) {
        float fold = 0.0;
        for (uint r = 0u; r < 32u; r += 1u) {
            fold = max(fold, red[r]);
        }
        // Keep the reduction live. `energy` is past the packed prefix.
        Dst.Store(energy * 4u + id * 4u, asuint(fold));
    }
}
