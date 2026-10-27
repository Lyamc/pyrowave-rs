//! Shared quant → analyze → finalize → resolve(target) → RDO pack.
//! One description for every backend. Integer steps follow the reference shaders
//! (`wavelet_quant.comp`, `analyze_rate_control.comp`, `analyze_rate_control_finalize.comp`,
//! `resolve_rate_control.comp`, `block_packing.comp`), not `tools/golden`.

use std::simd::Simd;

use crate::packet::{BlockMeta, FrameBits};
use crate::rdo_dwt::{f16_to_f32, f32_to_f16, forward_level_into, plane_from_u8_into};
use crate::rdo_scale::distortion_scale;
use crate::Chroma;

const LEVELS: usize = 5;
const Q_PLANES_OFFSET: i32 = 16;
const Q_PLANES_BITS: i32 = 4;
const QUANT_SCALE_OFFSET: i32 = 20;
const QUANT_SCALE_BITS: i32 = 4;
const IDENTITY_CODE: u32 = 6;

struct Band<'a> {
    w: i32,
    h: i32,
    level: usize,
    component: usize,
    band: usize,
    pixels: &'a [f32],
    off8: usize,
    stride8: i32,
    off32: usize,
    stride32: i32,
    n32x: i32,
    n32y: i32,
}

/// Planes and wavelet bands kept across frames. Callers pass the same scratch
/// into every encode so `Vec` capacity stays put. A second scratch holds the
/// next frame while the current one is packed.
pub struct Scratch {
    pub y: Vec<f32>,
    pub u: Vec<f32>,
    pub v: Vec<f32>,
    pub levels: Vec<Vec<[Vec<f32>; 4]>>,
}

impl Scratch {
    pub fn new() -> Self {
        let mut levels = Vec::with_capacity(3);
        for _ in 0..3 {
            let mut rows = Vec::with_capacity(LEVELS);
            for _ in 0..LEVELS {
                rows.push([Vec::new(), Vec::new(), Vec::new(), Vec::new()]);
            }
            levels.push(rows);
        }
        Self { y: Vec::new(), u: Vec::new(), v: Vec::new(), levels }
    }
}

/// One wavelet level. `dispatch` fills `out` (LL, HL, LH, HH). `submit_first`
/// may leave that level running; `finish_first` writes it before later levels
/// read the LL. The CPU wavelet finishes inside `finish_first`.
pub trait LevelWave {
    fn dispatch(
        &mut self,
        src: &[f32],
        res_w: i32,
        res_h: i32,
        aligned_w: i32,
        aligned_h: i32,
        dc: bool,
        out: &mut [Vec<f32>; 4],
    ) -> Result<(), String>;

    fn submit_first(
        &mut self,
        src: &[f32],
        res_w: i32,
        res_h: i32,
        aligned_w: i32,
        aligned_h: i32,
        dc: bool,
    ) -> Result<(), String>;

    fn finish_first(
        &mut self,
        src: &[f32],
        res_w: i32,
        res_h: i32,
        aligned_w: i32,
        aligned_h: i32,
        dc: bool,
        out: &mut [Vec<f32>; 4],
    ) -> Result<(), String>;
}

pub struct CpuWave;

impl LevelWave for CpuWave {
    fn dispatch(
        &mut self,
        src: &[f32],
        res_w: i32,
        res_h: i32,
        aligned_w: i32,
        aligned_h: i32,
        dc: bool,
        out: &mut [Vec<f32>; 4],
    ) -> Result<(), String> {
        forward_level_into(src, res_w, res_h, aligned_w, aligned_h, dc, out);
        Ok(())
    }

    fn submit_first(
        &mut self,
        _src: &[f32],
        _res_w: i32,
        _res_h: i32,
        _aligned_w: i32,
        _aligned_h: i32,
        _dc: bool,
    ) -> Result<(), String> {
        Ok(())
    }

    fn finish_first(
        &mut self,
        src: &[f32],
        res_w: i32,
        res_h: i32,
        aligned_w: i32,
        aligned_h: i32,
        dc: bool,
        out: &mut [Vec<f32>; 4],
    ) -> Result<(), String> {
        forward_level_into(src, res_w, res_h, aligned_w, aligned_h, dc, out);
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct Sub {
    code: u32,
    offset: u32,
    planes: u32,
    cost: [u16; 16],
    err: [f32; 16],
}

impl Sub {
    fn empty() -> Self {
        Self { code: 0, offset: 0, planes: 0, cost: [0; 16], err: [0.0; 16] }
    }
}

#[derive(Clone, Copy)]
struct Op {
    quant: i32,
    packed: u32,
}

pub fn encode_frame(
    scratch: &mut Scratch,
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: u32,
    height: u32,
    chroma: Chroma,
    target: u32,
    sequence: u32,
) -> Result<FrameBits, String> {
    encode_scratch(scratch, y, u, v, width, height, chroma, target, sequence, |src, rw, rh, alw, alh, dc, out| {
        forward_level_into(src, rw, rh, alw, alh, dc, out);
        Ok(())
    })
}

pub fn encode_scratch<F>(
    scratch: &mut Scratch,
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: u32,
    height: u32,
    chroma: Chroma,
    target: u32,
    sequence: u32,
    mut dispatch: F,
) -> Result<FrameBits, String>
where
    F: FnMut(&[f32], i32, i32, i32, i32, bool, &mut [Vec<f32>; 4]) -> Result<(), String>,
{
    prepare_planes(scratch, y, u, v, width, height, chroma)?;
    run_levels(scratch, width, height, chroma, false, &mut dispatch)?;
    encode_stored(&scratch.levels, width, height, chroma, target, sequence)
}

/// Wavelet `first` into `slot0`. Fill `slot1` with `second`'s planes, start that
/// frame's first level, pack `first`, then finish and pack `second`.
pub fn encode_pair_scratch<W: LevelWave>(
    slot0: &mut Scratch,
    slot1: &mut Scratch,
    y0: &[u8],
    u0: &[u8],
    v0: &[u8],
    y1: &[u8],
    u1: &[u8],
    v1: &[u8],
    width: u32,
    height: u32,
    chroma: Chroma,
    target: u32,
    seq0: u32,
    seq1: u32,
    wave: &mut W,
) -> Result<(FrameBits, FrameBits), String> {
    prepare_planes(slot0, y0, u0, v0, width, height, chroma)?;
    run_levels(slot0, width, height, chroma, false, &mut |src, rw, rh, alw, alh, dc, out| {
        wave.dispatch(src, rw, rh, alw, alh, dc, out)
    })?;
    prepare_planes(slot1, y1, u1, v1, width, height, chroma)?;
    let (aw, ah) = aligned_wh(width, height);
    let rw = width as i32;
    let rh = height as i32;
    wave.submit_first(&slot1.y, rw, rh, aw, ah, true)?;
    let first = encode_stored(&slot0.levels, width, height, chroma, target, seq0)?;
    {
        let Scratch { y, levels, .. } = &mut *slot1;
        wave.finish_first(y, rw, rh, aw, ah, true, &mut levels[0][0])?;
    }
    run_levels(slot1, width, height, chroma, true, &mut |src, rw, rh, alw, alh, dc, out| {
        wave.dispatch(src, rw, rh, alw, alh, dc, out)
    })?;
    let second = encode_stored(&slot1.levels, width, height, chroma, target, seq1)?;
    Ok((first, second))
}

pub fn prepare_planes(
    scratch: &mut Scratch,
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: u32,
    height: u32,
    chroma: Chroma,
) -> Result<(), String> {
    if width < 2 || height < 2 || width % 2 != 0 || height % 2 != 0 {
        return Err("width and height must be even and at least 2".into());
    }
    let chroma_444 = chroma == Chroma::Yuv444;
    let (cw, ch) = if chroma_444 {
        (width as usize, height as usize)
    } else {
        (width as usize / 2, height as usize / 2)
    };
    if y.len() < width as usize * height as usize || u.len() < cw * ch || v.len() < cw * ch {
        return Err("plane is short".into());
    }
    plane_from_u8_into(y, width as usize * height as usize, &mut scratch.y);
    plane_from_u8_into(u, cw * ch, &mut scratch.u);
    plane_from_u8_into(v, cw * ch, &mut scratch.v);
    Ok(())
}

fn aligned_wh(width: u32, height: u32) -> (i32, i32) {
    (align32(width as i32).max(128), align32(height as i32).max(128))
}

/// Run the reference level schedule. The next level reads `levels[level - 1][0]`
/// in place. `skip_luma0` is set when that level was already written by `finish_first`.
fn run_levels<F>(
    scratch: &mut Scratch,
    width: u32,
    height: u32,
    chroma: Chroma,
    skip_luma0: bool,
    dispatch: &mut F,
) -> Result<(), String>
where
    F: FnMut(&[f32], i32, i32, i32, i32, bool, &mut [Vec<f32>; 4]) -> Result<(), String>,
{
    let chroma_444 = chroma == Chroma::Yuv444;
    let (aw, ah) = aligned_wh(width, height);
    let Scratch { y, u, v, levels } = scratch;
    for comp in 0..3 {
        for level in 0..LEVELS {
            if skip_luma0 && comp == 0 && level == 0 {
                continue;
            }
            let comp_levels = &mut levels[comp];
            if level == 0 && comp != 0 && !chroma_444 {
                for band in &mut comp_levels[level] {
                    band.clear();
                }
                continue;
            }
            let from_plane = level == 0 || (!chroma_444 && comp != 0 && level == 1);
            let (rw, rh, alw, alh, dc) = if level == 0 {
                (width as i32, height as i32, aw, ah, true)
            } else if from_plane {
                ((width / 2) as i32, (height / 2) as i32, aw / 2, ah / 2, true)
            } else {
                let pw = aw >> level;
                let ph = ah >> level;
                (pw, ph, pw, ph, false)
            };
            let (head, tail) = comp_levels.split_at_mut(level);
            let dst = &mut tail[0];
            if from_plane {
                let src: &[f32] = if comp == 0 {
                    y
                } else if comp == 1 {
                    u
                } else {
                    v
                };
                dispatch(src, rw, rh, alw, alh, dc, dst)?;
            } else {
                let pw = aw >> level;
                let ph = ah >> level;
                let src = head[level - 1][0].as_slice();
                dispatch(src, pw, ph, pw, ph, false, dst)?;
            }
        }
    }
    Ok(())
}

/// Run the reference level schedule. `dispatch` writes LL, HL, LH, HH at `aligned/2`.
pub fn build_stored<F>(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: u32,
    height: u32,
    chroma: Chroma,
    mut dispatch: F,
) -> Result<Vec<Vec<[Vec<f32>; 4]>>, String>
where
    F: FnMut(&[f32], i32, i32, i32, i32, bool, &mut [Vec<f32>; 4]) -> Result<(), String>,
{
    let mut scratch = Scratch::new();
    prepare_planes(&mut scratch, y, u, v, width, height, chroma)?;
    run_levels(&mut scratch, width, height, chroma, false, &mut dispatch)?;
    Ok(scratch.levels)
}

pub fn encode_stored(
    stored: &[Vec<[Vec<f32>; 4]>],
    width: u32,
    height: u32,
    chroma: Chroma,
    target: u32,
    sequence: u32,
) -> Result<FrameBits, String> {
    let chroma_444 = chroma == Chroma::Yuv444;
    let aw = align32(width as i32).max(128);
    let ah = align32(height as i32).max(128);
    let mut bands = Vec::new();
    let mut count8 = 0usize;
    let mut count32 = 0usize;
    for level in (0..LEVELS).rev() {
        for comp in 0..3 {
            if level == 0 && comp != 0 && !chroma_444 {
                continue;
            }
            let band0 = if level == LEVELS - 1 { 0 } else { 1 };
            for band in band0..4 {
                let pixels = stored[comp][level][band].as_slice();
                let w = aw >> (level as i32 + 1);
                let h = ah >> (level as i32 + 1);
                let bx = (w + 7) / 8;
                let by = (h + 7) / 8;
                let n32x = (w + 31) / 32;
                let n32y = (h + 31) / 32;
                bands.push(Band {
                    w,
                    h,
                    level,
                    component: comp,
                    band,
                    pixels,
                    off8: count8,
                    stride8: bx,
                    off32: count32,
                    stride32: n32x,
                    n32x,
                    n32y,
                });
                count8 += (bx * by) as usize;
                count32 += (n32x * n32y) as usize;
            }
        }
    }
    if count32 >= 65536 {
        return Err("frame has too many 32x32 blocks for the 16-bit block index".into());
    }

    let mut subs = vec![Sub::empty(); count8];
    let mut payload: Vec<u8> = Vec::new();
    for band in &bands {
        let qr = pushed_quant_resolution(band.level as i32, band.component as i32, band.band as i32);
        let scale = distortion_scale(!chroma_444, band.level, band.component, band.band);
        quant_band(band, qr, scale, &mut subs, &mut payload);
    }

    let per_sub = per_subdivision(count32);
    let shamt = per_sub.trailing_zeros();
    let aligned = per_sub * 16;
    let mut savings = vec![0u32; 2048];
    let mut ops = vec![Op { quant: 0, packed: 0 }; 2048 * per_sub];
    let mut consumed = 0u32;
    for band in &bands {
        analyze_band(band, &subs, per_sub, shamt, aligned, &mut savings, &mut ops, &mut consumed);
    }
    finalize(&mut savings);
    let mut quant = vec![0i32; count32];
    let mut words_budget = target;
    if words_budget >= 8 {
        words_budget -= 8;
    }
    resolve(&savings, &ops, per_sub, consumed, (words_budget / 4) as i32, &mut quant);

    let mut blocks = vec![BlockMeta { bytes: Vec::new() }; count32];
    for band in &bands {
        let code = encode_quant(1.0 / quant_resolution(band.level as i32, band.component as i32, band.band as i32));
        pack_band(band, &subs, &payload, &quant, code, sequence & 7, &mut blocks);
    }
    Ok(FrameBits { width, height, chroma_444, blocks })
}

fn align32(v: i32) -> i32 {
    (v + 31) & !31
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_schedule_runs_thirteen_420_levels() {
        let y = vec![0u8; 8 * 8];
        let c = vec![128u8; 4 * 4];
        let mut calls = 0u32;
        let mut saw_dc = false;
        build_stored(&y, &c, &c, 8, 8, Chroma::Yuv420, |src, rw, rh, alw, alh, dc, out| {
            calls += 1;
            assert_eq!(src.len(), (rw as usize) * (rh as usize));
            if dc {
                saw_dc = true;
            }
            let n = ((alw / 2) * (alh / 2)) as usize;
            for band in out.iter_mut() {
                band.clear();
                band.resize(n, 0.0);
            }
            Ok(())
        })
        .expect("schedule");
        assert_eq!(calls, 13);
        assert!(saw_dc);
    }
}

fn quant_resolution(level: i32, component: i32, band: i32) -> f32 {
    let mut bits = 8;
    if band == 0 {
        bits += 2;
    } else if band < 3 {
        bits += 1;
    }
    bits += level;
    if component != 0 {
        bits -= 1;
    }
    ((1u32 << bits) as f32).min(4096.0)
}

fn encode_quant(scale: f32) -> u8 {
    let v = scale.to_bits();
    let mut e = ((v >> 23) & 0xff) as i32 - 127 - 4;
    let m = ((v >> 20) & 7) as i32;
    e = -e;
    ((e as u8) << 3) | (m as u8)
}

fn decode_quant(code: u8) -> f32 {
    let e = 4 - ((code >> 3) as i32);
    let m = (code & 7) as i32;
    let shifted = (8 + m) * (1 << (20 + e));
    (1.0 / (8.0 * 1024.0 * 1024.0)) * (shifted as f32)
}

fn pushed_quant_resolution(level: i32, component: i32, band: i32) -> f32 {
    let q = quant_resolution(level, component, band);
    1.0 / decode_quant(encode_quant(1.0 / q))
}

fn bex(v: u32, off: i32, bits: i32) -> u32 {
    if bits <= 0 {
        return 0;
    }
    (v >> off) & ((1u32 << bits) - 1)
}

fn bins(v: u32, ins: u32, off: i32, bits: i32) -> u32 {
    let mask = ((1u32 << bits) - 1) << off;
    (v & !mask) | ((ins << off) & mask)
}

fn find_msb(v: i32) -> i32 {
    if v <= 0 {
        -1
    } else {
        31 - (v as u32).leading_zeros() as i32
    }
}

fn frexp_exp(x: f32) -> i32 {
    let bits = x.abs().to_bits();
    let biased = ((bits >> 23) & 0xff) as i32;
    if biased == 0 {
        0
    } else {
        biased - 126
    }
}

fn texel(band: &Band, x: i32, y: i32) -> f32 {
    if x < 0 || y < 0 || x >= band.w || y >= band.h {
        0.0
    } else {
        band.pixels[(y * band.w + x) as usize]
    }
}

fn cluster_add_f(v: &mut [f32; 8]) {
    for shift in [1usize, 2, 4] {
        let old = *v;
        for lane in 0..8 {
            v[lane] += old[lane ^ shift];
        }
    }
}

fn cluster_max(v: &mut [f32; 8]) {
    for shift in [1usize, 2, 4] {
        let old = *v;
        for lane in 0..8 {
            v[lane] = v[lane].max(old[lane ^ shift]);
        }
    }
}

fn scan8(v: &mut [i32; 8]) {
    for shift in [1usize, 2, 4] {
        let old = *v;
        for lane in 0..8 {
            if lane >= shift {
                v[lane] += old[lane - shift];
            }
        }
    }
}

#[derive(Clone, Copy)]
struct LaneQ {
    tex: [[f32; 4]; 2],
    abs_i: [[i32; 4]; 2],
}

impl LaneQ {
    fn zero() -> Self {
        Self { tex: [[0.0; 4]; 2], abs_i: [[0; 4]; 2] }
    }
}

fn lane_texels(band: &Band, x0: i32, y0: i32, qr: f32) -> (LaneQ, f32) {
    let mut samples = [0.0f32; 8];
    let mut n = 0usize;
    for dx in [0i32, 2] {
        samples[n] = texel(band, x0 + dx, y0);
        samples[n + 1] = texel(band, x0 + dx, y0 + 1);
        samples[n + 2] = texel(band, x0 + dx + 1, y0);
        samples[n + 3] = texel(band, x0 + dx + 1, y0 + 1);
        n += 4;
    }
    let scaled = (Simd::<f32, 8>::from_array(samples) * Simd::splat(qr)).to_array();
    let mut lane = LaneQ::zero();
    let mut max_abs = 0.0f32;
    for part in 0..2 {
        for i in 0..4 {
            let s = scaled[part * 4 + i];
            lane.tex[part][i] = s;
            max_abs = max_abs.max(s.abs());
            // ivec4(float) truncates toward zero before abs.
            lane.abs_i[part][i] = (s as i32).abs();
        }
    }
    (lane, max_abs)
}

fn scale_lane(lane: &mut LaneQ, quant_scale: f32) -> f32 {
    let flat = [
        lane.tex[0][0],
        lane.tex[0][1],
        lane.tex[0][2],
        lane.tex[0][3],
        lane.tex[1][0],
        lane.tex[1][1],
        lane.tex[1][2],
        lane.tex[1][3],
    ];
    let scaled = (Simd::<f32, 8>::from_array(flat) * Simd::splat(quant_scale)).to_array();
    let mut max_abs = 0.0f32;
    for part in 0..2 {
        for i in 0..4 {
            let s = scaled[part * 4 + i];
            lane.tex[part][i] = s;
            lane.abs_i[part][i] = (s as i32).abs();
            max_abs = max_abs.max(s.abs());
        }
    }
    max_abs
}

fn square_error(lane: &LaneQ, q: i32, dist_scale: f32, inv_q2: f32) -> (f32, u32) {
    let mut err = 0.0f32;
    let mut sig = 0u32;
    for part in 0..2 {
        for i in 0..4 {
            let a = lane.tex[part][i].abs();
            let mut iv = (a * 2.0f32.powi(-q)).floor();
            if iv != 0.0 {
                sig += 1;
                iv += 0.5;
            }
            iv = (iv * 2.0f32.powi(q)).trunc();
            let d = a - iv;
            err += d * d;
        }
    }
    (err * dist_scale * inv_q2, sig)
}

fn quant_band(band: &Band, qr: f32, dist_scale: f32, subs: &mut [Sub], payload: &mut Vec<u8>) {
    let nx = (band.w + 31) / 32;
    let ny = (band.h + 31) / 32;
    let n8x = (band.w + 7) / 8;
    let n8y = (band.h + 7) / 8;
    for gy in 0..ny {
        for gx in 0..nx {
            for cluster in 0..16 {
                let bx = (cluster & 3) as i32;
                let by = (cluster >> 2) as i32;
                let block_x = 4 * gx + bx;
                let block_y = 4 * gy + by;
                if block_x >= n8x || block_y >= n8y {
                    continue;
                }
                let index = band.off8 + (block_y * band.stride8 + block_x) as usize;
                let mut lanes = [LaneQ::zero(); 8];
                let mut max_sub = [0.0f32; 8];
                for lane in 0..8u32 {
                    let (lx, ly) = {
                        let index = lane << 3;
                        let y = (index & 1) | (((index >> 3) & 3) << 1);
                        let x = ((index >> 1) & 3) | (((index >> 5) & 1) << 2);
                        (x as i32, y as i32)
                    };
                    let x0 = gx * 32 + bx * 8 + lx;
                    let y0 = gy * 32 + by * 8 + ly;
                    let (lq, m) = lane_texels(band, x0, y0, qr);
                    max_sub[lane as usize] = m;
                    lanes[lane as usize] = lq;
                }
                cluster_max(&mut max_sub);
                let max_wave = max_sub[0];
                let (quant_code, quant_scale) = if max_wave < 1.0 {
                    (IDENTITY_CODE, 1.0f32)
                } else {
                    let e = frexp_exp(max_wave - 0.25);
                    let target_max = 2.0f32.powi(e) - 0.25;
                    let inv_scale = max_wave / target_max;
                    let code = ((inv_scale - 0.25) * 8.0).ceil() as u32;
                    let qs = 1.0 / ((code as f32) / 8.0 + 0.25);
                    (code, qs)
                };
                let mut sub_max = [0.0f32; 8];
                for lane in 0..8 {
                    sub_max[lane] = scale_lane(&mut lanes[lane], quant_scale);
                }
                let mut scaled_max = max_wave * quant_scale;
                // max_wave was the cluster max before the scale; scaling is uniform.
                let _ = scaled_max;
                scaled_max = max_wave * quant_scale;
                let max_abs = scaled_max as i32;
                if max_abs == 0 {
                    subs[index] = Sub::empty();
                    continue;
                }
                let msb = find_msb(max_abs);
                let overall = qr * quant_scale;
                let inv_q = 1.0 / overall;
                let inv_q2 = inv_q * inv_q;
                let mut early = [0i32; 8];
                let mut plane_code = [0u32; 8];
                let mut quality = 0i32;
                let mut block4 = [0i32; 8];
                for lane in 0..8 {
                    block4[lane] = sub_max[lane] as i32;
                }
                // quality planes and msb are functions of the cluster max, identical on every lane.
                quality = if msb >= 3 { msb - 2 } else { 0 };
                for lane in 0..8 {
                    let mut b4 = block4[lane] >> 0;
                    let mut cost = if b4 > 0 { 1 } else { 0 };
                    if msb >= 3 {
                        cost = quality + 1;
                        b4 >>= quality;
                    }
                    let pc = find_msb(b4) + 1;
                    plane_code[lane] = pc as u32;
                    cost += pc;
                    early[lane] = cost;
                    block4[lane] = b4;
                }
                let mut sig = [0u32; 8];
                let mut err0 = [0.0f32; 8];
                for lane in 0..8 {
                    let (e, s) = square_error(&lanes[lane], 0, dist_scale, inv_q2);
                    err0[lane] = e;
                    sig[lane] = s;
                }
                cluster_add_u(&mut sig);
                let mut late0 = 0i32;
                {
                    let mut parts = [0i32; 8];
                    for lane in 0..8 {
                        parts[lane] = (early[lane] - 1).max(0);
                    }
                    cluster_add_i(&mut parts);
                    late0 = 8 * parts[0] + sig[0] as i32;
                }
                let mut scan = early;
                scan8(&mut scan);
                let total = scan[7];
                let base = payload.len() as u32;
                for lane in 0..8 {
                    scan[lane] -= early[lane];
                }
                let mut merged = [0u32; 8];
                for lane in 0..8 {
                    merged[lane] = plane_code[lane] << (lane * 2);
                }
                for shift in [1usize, 2, 4] {
                    let old = merged;
                    for lane in 0..8 {
                        merged[lane] |= old[lane ^ shift];
                    }
                }
                let mut code = (quality as u32) << Q_PLANES_OFFSET;
                code = bins(code, quant_code, QUANT_SCALE_OFFSET, QUANT_SCALE_BITS);
                code |= merged[0];
                subs[index].code = code;
                subs[index].offset = base;
                subs[index].planes = (msb + 1) as u32;
                subs[index].cost[0] = late0 as u16;
                subs[index].err[0] = 0.0;

                for q in 1..=msb {
                    let mut se = [0.0f32; 8];
                    let mut sg = [0u32; 8];
                    let mut late_parts = [0i32; 8];
                    for lane in 0..8 {
                        let (e, s) = square_error(&lanes[lane], q, dist_scale, inv_q2);
                        se[lane] = e;
                        sg[lane] = s;
                        let mut b4 = (sub_max[lane] as i32) >> q;
                        let mut cost = if b4 > 0 { 1 } else { 0 };
                        let mut m = msb - q;
                        if m >= 3 {
                            let qp = m - 2;
                            cost = qp + 1;
                            b4 >>= qp;
                            m = find_msb(b4);
                        } else {
                            m = find_msb(b4);
                        }
                        cost += m + 1;
                        late_parts[lane] = (cost - 1).max(0);
                        let _ = cost;
                    }
                    cluster_add_f(&mut se);
                    cluster_add_u(&mut sg);
                    cluster_add_i(&mut late_parts);
                    // Recompute late bits from the q-specific early cost, matching compute_quant_stats.
                    let mut early_q = [0i32; 8];
                    for lane in 0..8 {
                        let mut b4 = (sub_max[lane] as i32) >> q;
                        let mut cost = if b4 > 0 { 1 } else { 0 };
                        if msb - q >= 3 {
                            let qp = msb - q - 2;
                            cost = qp + 1;
                            b4 >>= qp;
                        }
                        cost += find_msb(b4) + 1;
                        early_q[lane] = cost;
                    }
                    let mut parts = [0i32; 8];
                    for lane in 0..8 {
                        parts[lane] = (early_q[lane] - 1).max(0);
                    }
                    cluster_add_i(&mut parts);
                    let late = 8 * parts[0] + sg[0] as i32;
                    if (q as usize) < 16 {
                        subs[index].err[q as usize] = f16_to_f32(f32_to_f16(se[0].min(60000.0)));
                        subs[index].cost[q as usize] = late as u16;
                    }
                }
                let mut full = [0.0f32; 8];
                for lane in 0..8 {
                    let mut e = 0.0f32;
                    for part in 0..2 {
                        for i in 0..4 {
                            let t = lanes[lane].tex[part][i];
                            e += t * t;
                        }
                    }
                    full[lane] = e * inv_q2;
                }
                cluster_add_f(&mut full);
                let drop_at = (msb + 1) as usize;
                if drop_at < 16 {
                    subs[index].err[drop_at] = f16_to_f32(f32_to_f16(full[0].min(60000.0)));
                    subs[index].cost[drop_at] = 0;
                }

                for lane in 0..8 {
                    if early[lane] <= 0 {
                        continue;
                    }
                    let lq = &lanes[lane];
                    let mut sign = 0u32;
                    for i in 0..4 {
                        if lq.tex[0][i] < 0.0 {
                            sign |= 1 << i;
                        }
                        if lq.tex[1][i] < 0.0 {
                            sign |= 1 << (4 + i);
                        }
                    }
                    payload.push(sign as u8);
                    let iters = quality + plane_code[lane] as i32;
                    let mut qbit = iters - 1;
                    while qbit >= 0 {
                        let mut byte = 0u32;
                        for i in 0..4 {
                            byte |= bex(lq.abs_i[0][i] as u32, qbit, 1) << i;
                            byte |= bex(lq.abs_i[1][i] as u32, qbit, 1) << (4 + i);
                        }
                        payload.push(byte as u8);
                        qbit -= 1;
                    }
                }
                let wrote = payload.len() as u32 - base;
                if wrote != total as u32 {
                    // Keep the offset stable even if a lane's early cost disagreed.
                    let _ = wrote;
                }
            }
        }
    }
}

fn cluster_add_u(v: &mut [u32; 8]) {
    for shift in [1usize, 2, 4] {
        let old = *v;
        for lane in 0..8 {
            v[lane] = v[lane].wrapping_add(old[lane ^ shift]);
        }
    }
}

fn cluster_add_i(v: &mut [i32; 8]) {
    for shift in [1usize, 2, 4] {
        let old = *v;
        for lane in 0..8 {
            v[lane] += old[lane ^ shift];
        }
    }
}

fn per_subdivision(num_blocks: usize) -> usize {
    let aligned = (num_blocks + 15) & !15;
    let per = (aligned / 16).next_power_of_two();
    per.max(1)
}

fn analyze_band(
    band: &Band,
    subs: &[Sub],
    per_sub: usize,
    shamt: u32,
    aligned: usize,
    savings: &mut [u32],
    ops: &mut [Op],
    consumed: &mut u32,
) {
    let n8x = (band.w + 7) / 8;
    let n8y = (band.h + 7) / 8;
    for by32 in 0..band.n32y {
        for bx32 in 0..band.n32x {
            let block_index = band.off32 + (by32 * band.stride32 + bx32) as usize;
            let mut dist = [0.0f32; 16];
            let mut cost = [0u32; 16];
            for i in 0..16 {
                for sub in 0..16 {
                    let lx = (sub & 3) as i32;
                    let ly = (sub >> 2) as i32;
                    let x8 = 4 * bx32 + lx;
                    let y8 = 4 * by32 + ly;
                    if x8 >= n8x || y8 >= n8y {
                        continue;
                    }
                    let s = &subs[band.off8 + (y8 * band.stride8 + x8) as usize];
                    let idx = i.min(s.planes as usize).min(15);
                    let mut c = u32::from(s.cost[idx]);
                    if c != 0 {
                        c += 24;
                    }
                    dist[sub] += s.err[idx];
                    cost[sub] = cost[sub].wrapping_add(c);
                }
                for shift in [1usize, 2, 4, 8] {
                    let od = dist;
                    let oc = cost;
                    for lane in 0..16 {
                        // Butterfly is per quant index, across the 16 subblocks.
                        // Running it inside the quant loop mixes indices. Do it after.
                        let _ = (od, oc, lane, shift);
                    }
                }
            }
            // The loop above added every subblock into dist[sub] for every i, which is wrong.
            // Recompute cleanly.
            dist = [0.0; 16];
            cost = [0; 16];
            for i in 0..16 {
                let mut dlane = [0.0f32; 16];
                let mut clane = [0u32; 16];
                for sub in 0..16 {
                    let lx = (sub & 3) as i32;
                    let ly = (sub >> 2) as i32;
                    let x8 = 4 * bx32 + lx;
                    let y8 = 4 * by32 + ly;
                    if x8 >= n8x || y8 >= n8y {
                        continue;
                    }
                    let s = &subs[band.off8 + (y8 * band.stride8 + x8) as usize];
                    let idx = (i as u32).min(s.planes).min(15) as usize;
                    let mut c = u32::from(s.cost[idx]);
                    if c != 0 {
                        c += 24;
                    }
                    dlane[sub] = s.err[idx];
                    clane[sub] = c;
                }
                for shift in [1usize, 2, 4, 8] {
                    let od = dlane;
                    let oc = clane;
                    for lane in 0..16 {
                        dlane[lane] += od[lane ^ shift];
                        clane[lane] = clane[lane].wrapping_add(oc[lane ^ shift]);
                    }
                }
                let mut c0 = clane[0];
                if c0 != 0 {
                    c0 += 64;
                }
                cost[i] = (c0 + 31) >> 5;
                dist[i] = dlane[0];
            }
            let mut buckets = [0u32; 16];
            let d0 = dist[0];
            let cbase = cost[0] as f32;
            for i in 0..16 {
                buckets[i] = bucket_of(dist[i], cost[i] as f32, d0, cbase);
            }
            buckets[0] = 0;
            let buckets = inclusive_max16(buckets);
            *consumed = consumed.wrapping_add(cost[0]);
            for i in 1..16 {
                let saving = cost[i - 1].wrapping_sub(cost[i]);
                if saving == 0 {
                    continue;
                }
                let subdivision = block_index >> shamt;
                let b = buckets[i] as usize;
                if b < 128 && subdivision < 16 {
                    savings[b * 16 + subdivision] = savings[b * 16 + subdivision].wrapping_add(saving);
                }
                let slot = block_index + b * aligned;
                if slot < ops.len() {
                    ops[slot] = Op {
                        quant: i as i32,
                        packed: ((block_index as u32) & 0xffff) | (saving << 16),
                    };
                }
            }
        }
    }
}

fn bucket_of(d: f32, cost: f32, d_base: f32, cost_base: f32) -> u32 {
    if cost == cost_base {
        return 0;
    }
    let ratio = (d - d_base).max(0.0) / (cost_base - cost);
    let index = 60.0 + 2.0 * ratio.log2();
    let shifted = index + 0.5;
    if !shifted.is_finite() || shifted <= 0.0 {
        0
    } else {
        shifted as u32
    }
}

fn inclusive_max16(mut v: [u32; 16]) -> [u32; 16] {
    for lane in 0..16 {
        v[lane] = v[lane].min(128 - 16 + lane as u32);
    }
    for shift in [1usize, 2, 4, 8] {
        let old = v;
        for lane in 0..16 {
            if lane >= shift {
                let up = old[lane - shift].wrapping_add(shift as u32);
                v[lane] = v[lane].max(up);
            }
        }
    }
    v
}

fn finalize(savings: &mut [u32]) {
    let mut v = [[0u32; 4]; 512];
    let mut scan = [0u32; 512];
    for i in 0..512 {
        let b = i * 4;
        v[i] = [savings[b], savings[b + 1], savings[b + 2], savings[b + 3]];
        v[i][1] = v[i][1].wrapping_add(v[i][0]);
        v[i][2] = v[i][2].wrapping_add(v[i][1]);
        v[i][3] = v[i][3].wrapping_add(v[i][2]);
        scan[i] = v[i][3];
    }
    let mut step = 1usize;
    while step < 256 {
        let mut up = [0u32; 512];
        for i in 0..512 {
            if i >= step {
                up[i] = scan[i - step];
            }
        }
        for i in 0..512 {
            for k in 0..4 {
                v[i][k] = v[i][k].wrapping_add(up[i]);
            }
            scan[i] = v[i][3];
        }
        step *= 2;
    }
    for i in 0..512 {
        let b = i * 4;
        savings[b] = v[i][0];
        savings[b + 1] = v[i][1];
        savings[b + 2] = v[i][2];
        savings[b + 3] = v[i][3];
    }
}

fn resolve(savings: &[u32], ops: &[Op], per_sub: usize, consumed: u32, target_words: i32, quant: &mut [i32]) {
    let required0 = consumed as i32 - target_words;
    for x in 0..2048 {
        let mut req = required0;
        if x != 0 {
            if savings[x] == savings[x - 1] {
                continue;
            }
            req -= savings[x - 1] as i32;
        } else if savings[0] == 0 {
            continue;
        }
        if req <= 0 {
            continue;
        }
        let req_u = req as u32;
        let mut total_saved = 0u32;
        let mut i = 0usize;
        while i < per_sub && total_saved < req_u {
            let mut saving = [0u32; 32];
            let mut blk = [0u32; 32];
            let mut qv = [0i32; 32];
            for t in 0..32 {
                let idx = i + t;
                if idx < per_sub {
                    let op = &ops[x * per_sub + idx];
                    saving[t] = op.packed >> 16;
                    blk[t] = op.packed & 0xffff;
                    qv[t] = op.quant;
                }
            }
            let mut incl = saving;
            for shift in [1usize, 2, 4, 8, 16] {
                let old = incl;
                for t in 0..32 {
                    if t >= shift {
                        incl[t] = incl[t].wrapping_add(old[t - shift]);
                    }
                }
            }
            for t in 0..32 {
                let sav = saving[t];
                if sav != 0 && total_saved.wrapping_add(incl[t]).wrapping_sub(sav) < req_u {
                    let b = blk[t] as usize;
                    if b < quant.len() {
                        quant[b] = quant[b].max(qv[t]);
                    }
                }
            }
            total_saved = total_saved.wrapping_add(incl[31]);
            i += 32;
        }
    }
}

fn quantize_code_word(mut control: u32, mut quant: i32) -> u32 {
    if quant != 0 && control != 0 {
        let mut q_bits = bex(control, Q_PLANES_OFFSET, Q_PLANES_BITS) as i32;
        let sub = q_bits.min(quant);
        q_bits -= sub;
        quant -= sub;
        if quant != 0 {
            quant = quant.min(3);
            let mut plane0 = control & 0x5555;
            let mut plane1 = (control & 0xaaaa) >> 1;
            let mut plane2 = plane0 & plane1;
            loop {
                plane0 = plane1;
                plane1 = plane2;
                plane2 = 0;
                quant -= 1;
                if quant == 0 {
                    break;
                }
            }
            plane0 &= !plane1;
            let new_word = plane0 | (plane1 << 1);
            control = bins(control, new_word, 0, 16);
        }
        control = bins(control, q_bits as u32, Q_PLANES_OFFSET, Q_PLANES_BITS);
    }
    control
}

fn required_size(control: u32) -> u32 {
    let q_bits = bex(control, Q_PLANES_OFFSET, Q_PLANES_BITS);
    let lsbs = control & 0x5555;
    let mut msbs = control & 0xaaaa;
    msbs |= msbs >> 1;
    lsbs.count_ones() + msbs.count_ones() + q_bits * 8
}

fn modify_quant_code(mut code: u32, quant: i32) -> u32 {
    let mut e = bex(code, 3, 5) as i32;
    e = (e - quant).max(0);
    code = bins(code, e as u32, 3, 5);
    code
}

fn pack_band(
    band: &Band,
    subs: &[Sub],
    payload: &[u8],
    quant: &[i32],
    quant_code: u8,
    sequence: u32,
    blocks: &mut [BlockMeta],
) {
    let n8x = (band.w + 7) / 8;
    let n8y = (band.h + 7) / 8;
    for by32 in 0..band.n32y {
        for bx32 in 0..band.n32x {
            let block_index = band.off32 + (by32 * band.stride32 + bx32) as usize;
            let q = quant[block_index];
            let mut active = [false; 16];
            let mut code = [0u32; 16];
            let mut plane_bytes = [0u32; 16];
            let mut sign_bits = [0u32; 16];
            let mut bits_meta = [0u32; 16];
            let mut meta_code = [0u32; 16];
            let mut meta_off = [0u32; 16];
            for sub in 0..16 {
                let lx = (sub & 3) as i32;
                let ly = (sub >> 2) as i32;
                let x8 = 4 * bx32 + lx;
                let y8 = 4 * by32 + ly;
                if x8 >= n8x || y8 >= n8y {
                    continue;
                }
                let s = &subs[band.off8 + (y8 * band.stride8 + x8) as usize];
                let idx = (q as u32).min(s.planes).min(15) as usize;
                let nb = u32::from(s.cost[idx]);
                meta_code[sub] = s.code;
                meta_off[sub] = s.offset;
                let cw = quantize_code_word(s.code, q);
                code[sub] = cw;
                active[sub] = (cw & 0xffff) != 0;
                plane_bytes[sub] = required_size(cw);
                sign_bits[sub] = nb.wrapping_sub(plane_bytes[sub] * 8);
                bits_meta[sub] = if nb != 0 { nb + 24 } else { 0 };
            }
            let mut ballot = 0u32;
            for sub in 0..16 {
                if active[sub] {
                    ballot |= 1 << sub;
                }
            }
            let mut total_bits = 0u32;
            for sub in 0..16 {
                total_bits = total_bits.wrapping_add(bits_meta[sub]);
            }
            let mut words = (total_bits + 31) / 32;
            if words != 0 {
                words += 2;
            }
            if words == 0 {
                continue;
            }
            let mut pkt = vec![0u8; (words as usize) * 4];
            let word0 = ballot | (words << 16) | (sequence << 28);
            let word1 = modify_quant_code(u32::from(quant_code), q) | ((block_index as u32) << 8);
            pkt[0..4].copy_from_slice(&word0.to_le_bytes());
            pkt[4..8].copy_from_slice(&word1.to_le_bytes());
            let nsub = ballot.count_ones();
            let mut plane_excl = [0u32; 16];
            let mut sign_excl = [0u32; 16];
            let mut plane_sum = 0u32;
            let mut sign_sum = 0u32;
            let mut running_p = 0u32;
            let mut running_s = 0u32;
            for sub in 0..16 {
                plane_excl[sub] = running_p;
                sign_excl[sub] = running_s;
                running_p = running_p.wrapping_add(plane_bytes[sub]);
                running_s = running_s.wrapping_add(sign_bits[sub]);
            }
            plane_sum = running_p;
            sign_sum = running_s;
            let mut bank = [0u32; 32];
            let mut compact = 0u32;
            for sub in 0..16 {
                if !active[sub] {
                    continue;
                }
                let hdr = compact;
                compact += 1;
                let at = 8 + 2 * hdr as usize;
                if at + 2 <= pkt.len() {
                    pkt[at..at + 2].copy_from_slice(&(code[sub] as u16).to_le_bytes());
                }
                let qat = 8 + 2 * nsub as usize + hdr as usize;
                if qat < pkt.len() {
                    pkt[qat] = (code[sub] >> 16) as u8;
                }
                let in_q = bex(meta_code[sub], Q_PLANES_OFFSET, Q_PLANES_BITS);
                let out_q = bex(code[sub], Q_PLANES_OFFSET, Q_PLANES_BITS);
                let mut input = meta_off[sub] as usize;
                let mut output = (4 * 0 + 3 * nsub + 8 + plane_excl[sub]) as usize;
                // global payload offset inside this packet is 0; planes start at 8+3N.
                let _ = output;
                output = (8 + 3 * nsub + plane_excl[sub]) as usize;
                let mut local_sign = sign_excl[sub];
                let mut pending = 0u32;
                let mut pending_mask = 0u32;
                for bit in (0..16).step_by(2) {
                    let mut out_planes = bex(code[sub], bit, 2) + out_q;
                    let mut in_planes = bex(meta_code[sub], bit, 2) + in_q;
                    if in_planes != 0 {
                        in_planes += 1;
                    }
                    let sign_plane = if input < payload.len() { u32::from(payload[input]) } else { 0 };
                    if out_planes != 0 {
                        let mut significant = 0u32;
                        let mut src_i = input + 1;
                        while out_planes > 0 {
                            let byte = if src_i < payload.len() { payload[src_i] } else { 0 };
                            significant |= u32::from(byte);
                            if output < pkt.len() {
                                pkt[output] = byte;
                            }
                            output += 1;
                            src_i += 1;
                            out_planes -= 1;
                        }
                        let mut sig = significant;
                        while sig != 0 {
                            let b = sig.trailing_zeros() as i32;
                            sig &= sig.wrapping_sub(1);
                            let out_bit = (local_sign & 31) as i32;
                            pending = bins(pending, bex(sign_plane, b, 1), out_bit, 1);
                            pending_mask = bins(pending_mask, 1, out_bit, 1);
                            if out_bit == 31 {
                                let word = (local_sign / 32) as usize;
                                if word < bank.len() {
                                    if pending_mask == !0 {
                                        bank[word] = pending;
                                    } else {
                                        bank[word] = (bank[word] & !pending_mask) | (pending & pending_mask);
                                    }
                                }
                                pending_mask = 0;
                            }
                            local_sign += 1;
                        }
                    }
                    input = input.wrapping_add(in_planes as usize);
                }
                if pending_mask != 0 {
                    let word = (local_sign / 32) as usize;
                    if word < bank.len() {
                        bank[word] = (bank[word] & !pending_mask) | (pending & pending_mask);
                    }
                }
            }
            let sign_bytes = (sign_sum + 7) / 8;
            let sign_base = (8 + 3 * nsub + plane_sum) as usize;
            let mut i = 0u32;
            while i < sign_bytes / 4 {
                let word = bank[i as usize];
                let at = sign_base + 4 * i as usize;
                if at + 4 <= pkt.len() {
                    pkt[at..at + 4].copy_from_slice(&word.to_le_bytes());
                }
                i += 1;
            }
            let mut j = sign_bytes & !3;
            while j < sign_bytes {
                let word = bank[(j / 4) as usize];
                let at = sign_base + j as usize;
                if at < pkt.len() {
                    pkt[at] = (word >> (8 * (j & 3))) as u8;
                }
                j += 1;
            }
            let _ = plane_sum;
            blocks[block_index].bytes = pkt;
        }
    }
}
