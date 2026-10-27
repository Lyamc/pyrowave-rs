//! Forward CDF 9/7 matching `reference/pyrowave/shaders/dwt.comp` at precision 1:
//! fp32 lifting, fp16 shared-memory stores, 32x32 tiles with a 4-texel apron.
//! Lifting steps run as `std::simd` pairs, the same two-wide vectors the shader packs.

use std::simd::Simd;

use crate::simd_dwt::{ALPHA, BETA, DELTA, GAMMA, INV_K, K};

const APRON: i32 = 4;
const APRON_HALF: i32 = 2;
const BLOCK: i32 = 32;
const BLOCK_HALF: i32 = 16;
const ROWS: usize = 20;
const COLS: usize = 41;

type V2 = Simd<f32, 2>;

pub fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;
    if exp == 255 {
        if mant == 0 {
            return sign | 0x7c00;
        }
        let mut nan = (mant >> 13) as u16;
        if nan == 0 {
            nan = 1;
        }
        return sign | 0x7c00 | nan;
    }
    if exp == 0 {
        return sign;
    }
    let mut e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mant | 0x0080_0000;
        let shift = (14 - e) as u32;
        let mut h = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        if rem > half || (rem == half && (h & 1) == 1) {
            h += 1;
        }
        return sign | (h as u16);
    }
    let mut h = (mant >> 13) as u16;
    let round = (mant >> 12) & 1;
    let sticky = mant & 0x0fff;
    if round == 1 && (sticky != 0 || (h & 1) == 1) {
        h += 1;
        if h == 0x400 {
            h = 0;
            e += 1;
            if e >= 31 {
                return sign | 0x7c00;
            }
        }
    }
    sign | ((e as u16) << 10) | h
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x03ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut m = mant;
            let mut e = 127 - 15 + 1;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            sign | (e << 23) | (m << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + (127 - 15)) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

fn pack2(v: V2) -> u32 {
    let a = v.to_array();
    u32::from(f32_to_f16(a[0])) | (u32::from(f32_to_f16(a[1])) << 16)
}

fn unpack2(bits: u32) -> V2 {
    V2::from_array([f16_to_f32(bits as u16), f16_to_f32((bits >> 16) as u16)])
}

fn unswizzle8x8(index: u32) -> (i32, i32) {
    let y = (index & 1) | (((index >> 3) & 3) << 1);
    let x = ((index >> 1) & 3) | (((index >> 5) & 1) << 2);
    (x as i32, y as i32)
}

/// Integer texel where `textureGather` starts, before the sampler wraps it.
/// `generate_mirror_uv` (`dwt.comp:28-37`) then `u = s * size`, `i0 = floor(u - 0.5)`
/// (Vulkan spec 20.3.8 and 20.4.1). Mirrored-repeat is applied to that integer, not
/// to the normalized coordinate: folding `s` first swaps the apron pair.
fn gather_raw(coord: i32, resolution: i32, aligned: i32) -> i32 {
    let mut c = coord;
    if c < 0 {
        c -= 1;
    }
    c += 1;
    let end = 2 * aligned - resolution;
    let past = c + 2 * (resolution - aligned) + 1;
    let mapped = if c >= end { past } else { c.min(resolution) };
    let inv = 1.0f32 / resolution as f32;
    let u = (mapped as f32) * inv * (resolution as f32);
    (u - 0.5).floor() as i32
}

fn wrap_mirror(mut i: i32, size: i32) -> i32 {
    if size <= 1 {
        return 0;
    }
    let n = size * 2;
    i %= n;
    if i < 0 {
        i += n;
    }
    if i >= size {
        i = n - 1 - i;
    }
    i
}

fn gather4(src: &[f32], w: i32, h: i32, aw: i32, ah: i32, x: i32, y: i32, dc: bool) -> [f32; 4] {
    let i0 = gather_raw(x, w, aw);
    let j0 = gather_raw(y, h, ah);
    let at = |ix: i32, iy: i32| -> f32 {
        let xx = wrap_mirror(ix, w) as usize;
        let yy = wrap_mirror(iy, h) as usize;
        // `uTexture` is a mediump sampler. The compiled SPIR-V decorates OpImageGather
        // with RelaxedPrecision, so the texel is rounded to fp16 before the highp
        // DC shift (`dwt.comp:45-51`). Wavelet inputs are already fp16 and stay put.
        let mut v = f16_to_f32(f32_to_f16(src[yy * w as usize + xx]));
        if dc {
            v -= 0.5;
        }
        v
    };
    [
        at(i0, j0),
        at(i0 + 1, j0),
        at(i0, j0 + 1),
        at(i0 + 1, j0 + 1),
    ]
}

struct Shared {
    cells: [[u32; COLS]; ROWS],
}

impl Shared {
    fn new() -> Self {
        Self { cells: [[0; COLS]; ROWS] }
    }
    fn store(&mut self, y: i32, x: i32, v: V2) {
        self.cells[y as usize][x as usize] = pack2(v);
    }
    fn load(&self, y: i32, x: i32) -> V2 {
        unpack2(self.cells[y as usize][x as usize])
    }
}

fn fma(coef: V2, factor: V2, acc: V2) -> V2 {
    let c = coef.to_array();
    let f = factor.to_array();
    let a = acc.to_array();
    // f32::mul_add is a single rounding, matching a contracted GLSL mad.
    V2::from_array([c[0].mul_add(f[0], a[0]), c[1].mul_add(f[1], a[1])])
}

fn lift(values: &mut [V2]) {
    let n = values.len() as i32;
    let alpha = V2::splat(ALPHA);
    let beta = V2::splat(BETA);
    let gamma = V2::splat(GAMMA);
    let delta = V2::splat(DELTA);
    // The GLSL lift is not `precise`, so the NVIDIA compiler contracts the
    // multiply-add. Match that fma; a separate mul then add flips f16 codes.
    let mut i = 1;
    while i < n - 1 {
        let sum = values[(i - 1) as usize] + values[(i + 1) as usize];
        values[i as usize] = fma(alpha, sum, values[i as usize]);
        i += 2;
    }
    i = 2;
    while i < n - 2 {
        let sum = values[(i - 1) as usize] + values[(i + 1) as usize];
        values[i as usize] = fma(beta, sum, values[i as usize]);
        i += 2;
    }
    i = 3;
    while i < n - 3 {
        let sum = values[(i - 1) as usize] + values[(i + 1) as usize];
        values[i as usize] = fma(gamma, sum, values[i as usize]);
        i += 2;
    }
    i = 4;
    while i < n - 4 {
        let sum = values[(i - 1) as usize] + values[(i + 1) as usize];
        values[i as usize] = fma(delta, sum, values[i as usize]);
        i += 2;
    }
}

struct Pending {
    y: i32,
    x: i32,
    v: V2,
}

fn xform8(shared: &mut Shared) {
    let mut pending = Vec::with_capacity(512);
    let inv_k = V2::splat(INV_K);
    let k = V2::splat(K);
    for local in 0..64u32 {
        let local_coord_x = 8 * (local % 4) as i32;
        let local_coord_y = (local / 4) as i32;
        let mut values = [V2::splat(0.0); 16];
        for i in 0..16 {
            values[i] = shared.load(local_coord_y, local_coord_x + i as i32);
        }
        lift(&mut values);
        let mut i = APRON_HALF;
        while i < 8 - APRON_HALF {
            let a = values[(2 * i) as usize] * inv_k;
            let b = values[(2 * i + 1) as usize] * k;
            let t0 = V2::from_array([a.to_array()[0], b.to_array()[0]]);
            let t1 = V2::from_array([a.to_array()[1], b.to_array()[1]]);
            let y_coord = (local_coord_x >> 1) + (i - APRON_HALF);
            pending.push(Pending { y: y_coord, x: 2 * local_coord_y, v: t0 });
            pending.push(Pending { y: y_coord, x: 2 * local_coord_y + 1, v: t1 });
            i += 1;
        }
    }
    for p in pending {
        shared.store(p.y, p.x, p.v);
    }
}

fn xform4(shared: &mut Shared) {
    let mut pending = Vec::with_capacity(256);
    let inv_k = V2::splat(INV_K);
    let k = V2::splat(K);
    for local in 0..32u32 {
        let local_coord_x = 4 * (local % 8) as i32;
        let local_coord_y = (local / 8) as i32 + BLOCK_HALF;
        let mut values = [V2::splat(0.0); 12];
        for i in 0..12 {
            values[i] = shared.load(local_coord_y, local_coord_x + i as i32);
        }
        lift(&mut values);
        let mut i = APRON_HALF;
        while i < 6 - APRON_HALF {
            let a = values[(2 * i) as usize] * inv_k;
            let b = values[(2 * i + 1) as usize] * k;
            let t0 = V2::from_array([a.to_array()[0], b.to_array()[0]]);
            let t1 = V2::from_array([a.to_array()[1], b.to_array()[1]]);
            let y_coord = (local_coord_x >> 1) + (i - APRON_HALF);
            pending.push(Pending { y: y_coord, x: 2 * local_coord_y, v: t0 });
            pending.push(Pending { y: y_coord, x: 2 * local_coord_y + 1, v: t1 });
            i += 1;
        }
    }
    for p in pending {
        shared.store(p.y, p.x, p.v);
    }
}

fn load_apron(shared: &mut Shared, src: &[f32], w: i32, h: i32, aw: i32, ah: i32, dc: bool, gx: i32, gy: i32) {
    let base_x = gx * BLOCK - APRON;
    let base_y = gy * BLOCK - APRON;
    for local in 0..64u32 {
        let (ux, uy) = unswizzle8x8(local);
        let lx = 2 * ux;
        let ly = 2 * uy;
        let g = |ox: i32, oy: i32| gather4(src, w, h, aw, ah, base_x + lx + ox, base_y + ly + oy, dc);
        let t0 = g(0, 0);
        let t1 = g(16, 0);
        let t2 = g(0, 16);
        let t3 = g(16, 16);
        let sy = ly >> 1;
        shared.store(sy, lx, V2::from_array([t0[0], t0[2]]));
        shared.store(sy, lx + 1, V2::from_array([t0[1], t0[3]]));
        shared.store(sy, lx + 16, V2::from_array([t1[0], t1[2]]));
        shared.store(sy, lx + 17, V2::from_array([t1[1], t1[3]]));
        shared.store(sy + 8, lx, V2::from_array([t2[0], t2[2]]));
        shared.store(sy + 8, lx + 1, V2::from_array([t2[1], t2[3]]));
        shared.store(sy + 8, lx + 16, V2::from_array([t3[0], t3[2]]));
        shared.store(sy + 8, lx + 17, V2::from_array([t3[1], t3[3]]));

        let ax = BLOCK + 2 * (local % 4) as i32;
        let ay = 2 * (local / 4) as i32;
        let tr = gather4(src, w, h, aw, ah, base_x + ax, base_y + ay, dc);
        shared.store(ay >> 1, ax, V2::from_array([tr[0], tr[2]]));
        shared.store(ay >> 1, ax + 1, V2::from_array([tr[1], tr[3]]));

        let bx = 2 * (local % 16) as i32;
        let by = BLOCK + 2 * (local / 16) as i32;
        let bl = gather4(src, w, h, aw, ah, base_x + bx, base_y + by, dc);
        shared.store(by >> 1, bx, V2::from_array([bl[0], bl[2]]));
        shared.store(by >> 1, bx + 1, V2::from_array([bl[1], bl[3]]));

        if local < 16 {
            let cx = BLOCK + 2 * (local % 4) as i32;
            let cy = BLOCK + 2 * (local / 4) as i32;
            let br = gather4(src, w, h, aw, ah, base_x + cx, base_y + cy, dc);
            shared.store(cy >> 1, cx, V2::from_array([br[0], br[2]]));
            shared.store(cy >> 1, cx + 1, V2::from_array([br[1], br[3]]));
        }
    }
}

/// One forward level. `src` is `res_w * res_h` samples (UNORM 0..1, or a previous LL).
/// Output bands are `aligned/2` on each axis: layer 0 LL, 1 HL, 2 LH, 3 HH.
pub fn forward_level(src: &[f32], res_w: i32, res_h: i32, aligned_w: i32, aligned_h: i32, dc: bool) -> [Vec<f32>; 4] {
    let ow = aligned_w / 2;
    let oh = aligned_h / 2;
    let n = (ow * oh) as usize;
    let mut bands = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for b in &mut bands {
        b.resize(n, 0.0);
    }
    let groups_x = (aligned_w + 31) / 32;
    let groups_y = (aligned_h + 31) / 32;
    for gy in 0..groups_y {
        for gx in 0..groups_x {
            let mut shared = Shared::new();
            load_apron(&mut shared, src, res_w, res_h, aligned_w, aligned_h, dc, gx, gy);
            xform8(&mut shared);
            xform4(&mut shared);
            xform8(&mut shared);
            for local in 0..64u32 {
                let (lx, ly0) = unswizzle8x8(local);
                let mut y = ly0;
                while y < BLOCK_HALF {
                    let mut x = lx * 2;
                    while x < BLOCK {
                        let v0 = shared.load(y, x).to_array();
                        let v1 = shared.load(y, x + 1).to_array();
                        let img_x = gx * BLOCK_HALF + (x >> 1);
                        let img_y = gy * BLOCK_HALF + y;
                        if img_x >= 0 && img_y >= 0 && img_x < ow && img_y < oh {
                            let at = (img_y * ow + img_x) as usize;
                            bands[0][at] = v0[0];
                            bands[2][at] = v0[1];
                            bands[1][at] = v1[0];
                            bands[3][at] = v1[1];
                        }
                        x += 16;
                    }
                    y += 8;
                }
            }
        }
    }
    note_level("cpu", &bands);
    bands
}

pub(crate) fn note_level(tag: &str, bands: &[Vec<f32>; 4]) {
    if std::env::var_os("PYRO_DWT_STAT").is_none() {
        return;
    }
    static SEEN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let i = SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if i > 1 {
        return;
    }
    let ll = &bands[0];
    let n = ll.len();
    let at = |idx: usize| ll.get(idx).copied().unwrap_or(0.0).to_bits();
    let mid = n / 2;
    let mut xor = 0u32;
    for band in bands {
        for v in band {
            xor ^= v.to_bits();
        }
    }
    let mut row = String::new();
    for k in 0..32 {
        row.push_str(&format!("{:08x}", at(k)));
        if k + 1 != 32 {
            row.push(' ');
        }
    }
    eprintln!(
        "STAT {tag} i={i} n={n} first={:08x} mid={:08x} at100={:08x} xor={xor:08x}",
        at(0),
        at(mid),
        at(100),
    );
    eprintln!("ROW {tag} i={i} {row}");
    if i <= 1 {
        if let Ok(dir) = std::env::var("PYRO_DWT_DUMP") {
            let mut bytes = Vec::new();
            for band in bands {
                for v in band {
                    bytes.extend_from_slice(&v.to_bits().to_le_bytes());
                }
            }
            let path = format!("{dir}-{tag}-{i}.bin");
            eprintln!("DUMP {path} bytes={}", bytes.len());
            let _ = std::fs::write(&path, bytes);
        }
    }
}

pub fn plane_from_u8(bytes: &[u8], n: usize) -> Vec<f32> {
    bytes[..n].iter().map(|p| f32::from(*p) / 255.0).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_roundtrip_ones() {
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(-2.0), 0xc000);
        assert!((f16_to_f32(f32_to_f16(0.1)) - 0.1).abs() < 1e-4);
    }

    #[test]
    fn half_matches_f16c_rne() {
        if !std::is_x86_feature_detected!("f16c") {
            return;
        }
        let samples = [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            0.5,
            -2.0,
            65504.0,
            65520.0,
            1.0 / 255.0,
            128.0 / 255.0,
            200.0 / 255.0 - 0.5,
            1.0e-8,
            2.0f32.powi(-25),
            0.1,
            -0.1,
            1024.5,
        ];
        for &v in &samples {
            let hw = unsafe {
                let packed = std::arch::x86_64::_mm_cvtps_ph(std::arch::x86_64::_mm_set_ss(v), 0);
                std::arch::x86_64::_mm_extract_epi16(packed, 0) as u16
            };
            assert_eq!(f32_to_f16(v), hw, "f16 bits differ for {v}");
        }
        let mut mismatches = 0u32;
        for bits in 0..65536u32 {
            let v = f32::from_bits(bits.wrapping_mul(0x9E37_79B9));
            if !v.is_finite() {
                continue;
            }
            let hw = unsafe {
                let packed = std::arch::x86_64::_mm_cvtps_ph(std::arch::x86_64::_mm_set_ss(v), 0);
                std::arch::x86_64::_mm_extract_epi16(packed, 0) as u16
            };
            if f32_to_f16(v) != hw {
                mismatches += 1;
                if mismatches <= 8 {
                    eprintln!("f16 mismatch bits={:08x} soft={:04x} hw={:04x}", v.to_bits(), f32_to_f16(v), hw);
                }
            }
        }
        for k in 0..256u32 {
            let v = k as f32 / 255.0;
            let hw = unsafe {
                let packed = std::arch::x86_64::_mm_cvtps_ph(std::arch::x86_64::_mm_set_ss(v), 0);
                std::arch::x86_64::_mm_extract_epi16(packed, 0) as u16
            };
            assert_eq!(f32_to_f16(v), hw, "unorm {k}");
        }
        assert_eq!(mismatches, 0);
    }

    #[test]
    fn constant_high_bands_stay_small() {
        let w = 64i32;
        let h = 64i32;
        let src = vec![128.0f32 / 255.0; (w * h) as usize];
        let bands = forward_level(&src, w, h, 128, 128, true);
        let mut peak = 0.0f32;
        for b in 1..4 {
            for v in &bands[b] {
                peak = peak.max(v.abs());
            }
        }
        assert!(peak < 1.0e-3, "high-band peak {peak}");
    }

    #[test]
    fn mirror_formulas_on_apron() {
        fn mapped(mut c: i32, resolution: i32, aligned: i32) -> i32 {
            if c < 0 {
                c -= 1;
            }
            c += 1;
            let end = 2 * aligned - resolution;
            let past = c + 2 * (resolution - aligned) + 1;
            if c >= end { past } else { c.min(resolution) }
        }
        fn vk_wrap(i: i32, size: i32) -> i32 {
            let m = i.rem_euclid(size * 2);
            let t = m - size;
            let mirrored = if t >= 0 { t } else { -(1 + t) };
            (size - 1) - mirrored
        }
        let mut differ = 0u32;
        for &(res, aligned) in &[(64, 128), (720, 736), (1280, 1280), (32, 64), (360, 368)] {
            for c in -8..aligned + 8 {
                let m = mapped(c, res, aligned);
                let inv = 1.0f32 / res as f32;
                let u = (m as f32) * inv * (res as f32);
                let i0_raw = (u - 0.5).floor() as i32;
                let spec = [vk_wrap(i0_raw, res), vk_wrap(i0_raw + 1, res)];
                let rust = {
                    let i0 = gather_raw(c, res, aligned);
                    [wrap_mirror(i0, res), wrap_mirror(i0 + 1, res)]
                };
                if spec != rust {
                    differ += 1;
                    if differ <= 12 {
                        eprintln!("res={res} c={c} mapped={m} i0_raw={i0_raw} spec={spec:?} rust={rust:?}");
                    }
                }
            }
        }
        eprintln!("differ={differ}");
        assert_eq!(differ, 0);
    }
}
