//! CPU-side submission for the cs_5_0 DWT, quant, and pack shaders.
//! `d3d11::encode` uploads these bytes and dispatches this list. The shapes
//! match `src/dwt.cpp` (`PushConstants`, 192-byte batch cbuffer, Texture2DArray
//! planes and bands) and `src/quant.cpp` / `src/frame.cpp` (160-byte quant
//! push, 48-byte pack jobs).

/// `DXGI_FORMAT_R8_UNORM`. Input planes.
pub const FORMAT_R8_UNORM: u32 = 61;
/// `DXGI_FORMAT_R32_FLOAT`. Band UAVs and coefficient buffers.
pub const FORMAT_R32_FLOAT: u32 = 41;
/// `DXGI_FORMAT_R32_UINT`. Quant and pack byte-address views in the C++ port.
pub const FORMAT_R32_UINT: u32 = 42;
/// `DXGI_FORMAT_R32_TYPELESS`. Raw coefficient and job views.
pub const FORMAT_R32_TYPELESS: u32 = 39;
/// `DXGI_FORMAT_R32_SINT`. Quant-code buffer. Stays zero.
pub const FORMAT_R32_SINT: u32 = 43;

/// `D3D11_SRV_DIMENSION_TEXTURE2DARRAY`.
pub const SRV_DIM_TEXTURE2DARRAY: u32 = 5;
/// `D3D11_SRV_DIMENSION_BUFFEREX`.
pub const SRV_DIM_BUFFEREX: u32 = 11;
/// `D3D11_UAV_DIMENSION_TEXTURE2DARRAY`.
pub const UAV_DIM_TEXTURE2DARRAY: u32 = 5;
/// `D3D11_UAV_DIMENSION_BUFFER`.
pub const UAV_DIM_BUFFER: u32 = 1;

pub const DWT_RECORD_BYTES: usize = 64;
pub const DWT_PUSH_BYTES: usize = 192;
pub const QUANT_PUSH_BYTES: usize = 160;
pub const PACK_JOB_BYTES: usize = 48;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureBinding {
    pub width: u32,
    pub height: u32,
    pub format: u32,
    pub array_size: u32,
    pub srv_dimension: u32,
    pub uav_dimension: u32,
    pub srv_array_size: u32,
    pub uav_array_size: u32,
}

/// R8 plane, one array slice, shader resource is a Texture2DArray.
/// `dwt.cpp` `dwt_upload_planes`.
pub fn plane_texture_desc(width: u32, height: u32) -> TextureBinding {
    TextureBinding {
        width,
        height,
        format: FORMAT_R8_UNORM,
        array_size: 1,
        srv_dimension: SRV_DIM_TEXTURE2DARRAY,
        uav_dimension: 0,
        srv_array_size: 1,
        uav_array_size: 0,
    }
}

/// Four-layer R32_FLOAT band. UAV and SRV are Texture2DArray.
/// `dwt.cpp` `make_band`.
pub fn band_texture_desc(width: u32, height: u32) -> TextureBinding {
    TextureBinding {
        width,
        height,
        format: FORMAT_R32_FLOAT,
        array_size: 4,
        srv_dimension: SRV_DIM_TEXTURE2DARRAY,
        uav_dimension: UAV_DIM_TEXTURE2DARRAY,
        srv_array_size: 4,
        uav_array_size: 4,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DwtSlot {
    pub component: u8,
    pub level: u8,
    pub from_plane: bool,
    /// Previous level when `from_plane` is false. Ignored for a plane.
    pub src_level: u8,
    pub src_component: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DwtBatch {
    pub slots: Vec<DwtSlot>,
    /// 192 bytes. Record `z` is `dwt.hlsl` `jobw[z*4 .. z*4+3]`.
    pub push: Vec<u8>,
    pub groups_x: u32,
    pub groups_y: u32,
    pub groups_z: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuantDispatch {
    pub component: u8,
    pub level: u8,
    /// 160 bytes, `quant.hlsl` `uint4 w[10]`.
    pub push: Vec<u8>,
    pub groups_x: u32,
    pub groups_y: u32,
    pub groups_z: u32,
}

#[derive(Clone, Debug)]
pub struct Submission {
    pub aligned_w: i32,
    pub aligned_h: i32,
    pub plane_size: [(u32, u32); 3],
    pub dwt: Vec<DwtBatch>,
    pub quant: Vec<QuantDispatch>,
    /// Concatenated 48-byte pack jobs. Uploaded to `t3` before the one pack dispatch.
    pub pack_jobs: Vec<u8>,
    pub pack_groups_x: u32,
    pub pack_groups_y: u32,
    pub pack_groups_z: u32,
    pub total8: i32,
    pub total32: i32,
}

pub fn submission_plan(width: u32, height: u32, chroma_444: bool) -> Result<Submission, String> {
    if width < 2 || height < 2 || width % 2 != 0 || height % 2 != 0 {
        return Err("width and height must be even and at least 2".into());
    }
    let aw = align_dim(width);
    let ah = align_dim(height);
    let pw = [
        width,
        if chroma_444 { width } else { width / 2 },
        if chroma_444 { width } else { width / 2 },
    ];
    let ph = [
        height,
        if chroma_444 { height } else { height / 2 },
        if chroma_444 { height } else { height / 2 },
    ];
    let disps = dwt_dispatches(width, height, aw, ah, chroma_444, &pw, &ph);
    if disps.is_empty() {
        return Err("no dwt dispatches".into());
    }
    let masks = layer_masks(width, aw, ah, chroma_444);
    let dwt = batch_dispatches(&disps, &masks, chroma_444);
    let (quant, pack_jobs, pack_gx, pack_gy, total8, total32) =
        quant_and_pack(width, aw, ah, chroma_444);
    if quant.is_empty() || pack_jobs.is_empty() || total32 <= 0 {
        return Err("no quant or pack jobs".into());
    }
    let pack_groups_z = (pack_jobs.len() / PACK_JOB_BYTES) as u32;
    Ok(Submission {
        aligned_w: aw,
        aligned_h: ah,
        plane_size: [(pw[0], ph[0]), (pw[1], ph[1]), (pw[2], ph[2])],
        dwt,
        quant,
        pack_jobs,
        pack_groups_x: pack_gx,
        pack_groups_y: pack_gy,
        pack_groups_z,
        total8,
        total32,
    })
}

fn align_dim(v: u32) -> i32 {
    let a = ((v + 31) & !31) as i32;
    if a < 128 {
        128
    } else {
        a
    }
}

#[derive(Clone, Copy)]
struct Disp {
    component: u8,
    level: u8,
    from_plane: bool,
    src_component: u8,
    src_level: u8,
    rx: i32,
    ry: i32,
    ax: i32,
    ay: i32,
    dc: u32,
    gx: u32,
    gy: u32,
    coeff_pitch: u32,
    src_pitch: u32,
}

fn dwt_dispatches(
    width: u32,
    height: u32,
    aw: i32,
    ah: i32,
    chroma_444: bool,
    pw: &[u32; 3],
    ph: &[u32; 3],
) -> Vec<Disp> {
    let mut out = Vec::new();
    let mut push = |component: u8, level: u8, from_plane: bool, src_c: u8, src_l: u8, rx: i32, ry: i32, ax: i32, ay: i32, dc: bool, src_pitch: u32| {
        if rx <= 0 || ry <= 0 || ax <= 0 || ay <= 0 {
            return;
        }
        let bw = aw >> (level as i32 + 1);
        let pitch = (bw + 31) & !31;
        out.push(Disp {
            component,
            level,
            from_plane,
            src_component: src_c,
            src_level: src_l,
            rx,
            ry,
            ax,
            ay,
            dc: u32::from(dc),
            gx: ((ax + 31) / 32) as u32,
            gy: ((ay + 31) / 32) as u32,
            coeff_pitch: pitch as u32,
            src_pitch,
        });
    };

    if chroma_444 {
        for c in 0..3u8 {
            push(c, 0, true, c, 0, pw[c as usize] as i32, ph[c as usize] as i32, aw, ah, true, pw[c as usize]);
        }
    } else {
        push(0, 0, true, 0, 0, width as i32, height as i32, aw, ah, true, width);
    }
    for level in 1..5u8 {
        for c in 0..3u8 {
            let chroma_l1 = !chroma_444 && c > 0 && level == 1;
            if chroma_l1 {
                let ax = aw >> 1;
                let ay = ah >> 1;
                push(
                    c,
                    level,
                    true,
                    c,
                    0,
                    pw[c as usize] as i32,
                    ph[c as usize] as i32,
                    ax,
                    ay,
                    true,
                    pw[c as usize],
                );
            } else if chroma_444 || c == 0 || level > 1 {
                let prev_w = aw >> level as i32;
                let prev_h = ah >> level as i32;
                push(
                    c,
                    level,
                    false,
                    c,
                    level - 1,
                    prev_w,
                    prev_h,
                    prev_w,
                    prev_h,
                    false,
                    prev_w as u32,
                );
            }
        }
    }
    out
}

/// Bits 0..3 are coefficient layers. Bit 4 writes the LL texture.
/// `dwt.cpp` `layer_mask`: every level still runs, and only the leading
/// 60 (720p) or 96 (1080p) 32×32 blocks store detail coefficients.
fn layer_masks(width: u32, aw: i32, ah: i32, chroma_444: bool) -> [[u32; 5]; 3] {
    let mut mask = [[0u32; 5]; 3];
    if chroma_444 {
        return mask;
    }
    let keep = keep_blocks(width);
    let mut count32 = 0i32;
    for level in (0..5).rev() {
        for c in 0..3 {
            if level == 0 && c > 0 {
                continue;
            }
            let bw = aw >> (level + 1);
            let bh = ah >> (level + 1);
            if bw <= 0 || bh <= 0 {
                continue;
            }
            let blocks = ((bw + 31) / 32) * ((bh + 31) / 32);
            let first = if level == 4 { 0 } else { 1 };
            for band in first..4 {
                if count32 < keep {
                    mask[c][level] |= 1 << band;
                }
                count32 += blocks;
            }
        }
    }
    for c in 0..3 {
        for level in 0..4 {
            if level == 0 && c > 0 {
                continue;
            }
            let bw = aw >> (level + 2);
            let bh = ah >> (level + 2);
            if bw > 0 && bh > 0 {
                mask[c][level] |= 16;
            }
        }
    }
    mask
}

fn dwt_record(d: &Disp, slot: u32, mask: u32, chroma_444: bool) -> [u8; DWT_RECORD_BYTES] {
    let mut w = [0u32; 16];
    w[0] = d.rx as u32;
    w[1] = d.ry as u32;
    if chroma_444 {
        w[2] = (1.0f32 / d.rx as f32).to_bits();
        w[3] = (1.0f32 / d.ry as f32).to_bits();
    }
    w[4] = d.ax as u32;
    w[5] = d.ay as u32;
    w[6] = d.dc;
    w[7] = 1.0f32.to_bits();
    if chroma_444 {
        w[8] = 0;
        w[9] = d.gx;
        w[10] = d.gx.saturating_mul(d.gy).saturating_mul(20 * 64);
    } else {
        // `fill_pc`: phase and slot_stride are the batch slot.
        w[8] = slot;
        w[9] = d.gx;
        w[10] = slot;
    }
    w[11] = d.coeff_pitch;
    w[12] = d.src_pitch;
    w[13] = 0;
    // `dwt.cpp` fill_pc sends a zero mask as 0xF. The shader turns 0 into 0x1F,
    // which also stores the LL texture on levels the prefix does not keep.
    w[14] = if chroma_444 {
        0
    } else if mask == 0 {
        0xF
    } else {
        mask
    };
    w[15] = 0;
    let mut out = [0u8; DWT_RECORD_BYTES];
    for (i, word) in w.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    out
}

fn batch_dispatches(disps: &[Disp], masks: &[[u32; 5]; 3], chroma_444: bool) -> Vec<DwtBatch> {
    let groups: Vec<Vec<usize>> = if !chroma_444 && disps.len() == 13 {
        // `dwt.cpp` batches: L0 Y, then each full level Y/Cb/Cr.
        vec![
            vec![0],
            vec![1, 2, 3],
            vec![4, 5, 6],
            vec![7, 8, 9],
            vec![10, 11, 12],
        ]
    } else {
        (0..disps.len()).map(|i| vec![i]).collect()
    };
    let mut batches = Vec::new();
    for group in groups {
        let mut push = vec![0u8; DWT_PUSH_BYTES];
        let mut slots = Vec::new();
        let head = &disps[group[0]];
        for (s, di) in group.iter().enumerate() {
            let d = &disps[*di];
            let mask = masks[d.component as usize][d.level as usize];
            let rec = dwt_record(d, s as u32, mask, chroma_444);
            let at = s * DWT_RECORD_BYTES;
            push[at..at + DWT_RECORD_BYTES].copy_from_slice(&rec);
            slots.push(DwtSlot {
                component: d.component,
                level: d.level,
                from_plane: d.from_plane,
                src_level: d.src_level,
                src_component: d.src_component,
            });
        }
        batches.push(DwtBatch {
            slots,
            push,
            groups_x: head.gx.max(1),
            groups_y: head.gy.max(1),
            groups_z: group.len() as u32,
        });
    }
    batches
}

struct BandRef {
    level: u8,
    component: u8,
    band: u8,
    w: i32,
    h: i32,
    pitch: i32,
    off8: i32,
    stride8: i32,
    off32: i32,
    stride32: i32,
}

fn keep_blocks(width: u32) -> i32 {
    if width >= 1600 { 96 } else { 60 }
}

fn quant_and_pack(width: u32, aw: i32, ah: i32, chroma_444: bool) -> (Vec<QuantDispatch>, Vec<u8>, u32, u32, i32, i32) {
    let mut bands = Vec::new();
    let mut count8 = 0i32;
    let mut count32 = 0i32;
    let mut seen32 = 0i32;
    let keep = if chroma_444 { i32::MAX } else { keep_blocks(width) };
    for level in (0..5).rev() {
        for component in 0..3 {
            if level == 0 && component > 0 && !chroma_444 {
                continue;
            }
            let w = aw >> (level + 1);
            let h = ah >> (level + 1);
            if w <= 0 || h <= 0 {
                continue;
            }
            let pitch = (w + 31) & !31;
            let b8x = (w + 7) / 8;
            let b8y = (h + 7) / 8;
            let b32x = (w + 31) / 32;
            let b32y = (h + 31) / 32;
            let blocks32 = b32x * b32y;
            let first = if level == 4 { 0 } else { 1 };
            for band in first..4 {
                if seen32 >= keep {
                    seen32 += blocks32;
                    continue;
                }
                seen32 += blocks32;
                bands.push(BandRef {
                    level: level as u8,
                    component: component as u8,
                    band: band as u8,
                    w,
                    h,
                    pitch,
                    off8: count8,
                    stride8: b8x,
                    off32: count32,
                    stride32: b32x,
                });
                count8 += b8x * b8y;
                count32 += b32x * b32y;
            }
        }
    }

    let mut quant = Vec::new();
    let mut i = 0;
    while i < bands.len() {
        let b0 = &bands[i];
        let mut words = [0u32; 40];
        words[0] = b0.w as u32;
        words[1] = b0.h as u32;
        words[2] = ((b0.w + 7) / 8) as u32;
        words[3] = ((b0.h + 7) / 8) as u32;
        words[4] = (1.0f32 / b0.w as f32).to_bits();
        words[5] = (1.0f32 / b0.h as f32).to_bits();
        words[6] = b0.pitch as u32;
        let mut n = 0usize;
        while i < bands.len() && n < 4 {
            let b = &bands[i];
            if b.w != b0.w || b.h != b0.h || b.level != b0.level || b.component != b0.component {
                break;
            }
            let o = 8 + n * 8;
            words[o] = b.band as u32;
            words[o + 1] = quant_resolution(b.level as i32, b.component as i32, b.band as i32).to_bits();
            words[o + 2] = b.off8 as u32;
            words[o + 3] = b.stride8 as u32;
            words[o + 4] = (1.0f32 / 256.0f32).to_bits();
            n += 1;
            i += 1;
        }
        words[7] = n as u32;
        let mut push = vec![0u8; QUANT_PUSH_BYTES];
        for (k, word) in words.iter().enumerate() {
            push[k * 4..k * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        quant.push(QuantDispatch {
            component: b0.component,
            level: b0.level,
            push,
            groups_x: ((b0.w + 31) / 32).max(1) as u32,
            groups_y: ((b0.h + 31) / 32).max(1) as u32,
            groups_z: n as u32,
        });
    }

    let mut pack_jobs = Vec::new();
    let mut pack_gx = 1u32;
    let mut pack_gy = 1u32;
    for b in &bands {
        let b32x = (b.w + 31) / 32;
        let b32y = (b.h + 31) / 32;
        let b8x = (b.w + 7) / 8;
        let b8y = (b.h + 7) / 8;
        let fields = [
            b.w as u32,
            b.h as u32,
            b32x as u32,
            b32y as u32,
            b8x as u32,
            b8y as u32,
            quant_code(b.level as i32, b.component as i32, b.band as i32),
            1u32, // sequence stays 1 so two encodes match
            b.off32 as u32,
            b.stride32 as u32,
            b.off8 as u32,
            b.stride8 as u32,
        ];
        for field in fields {
            pack_jobs.extend_from_slice(&field.to_le_bytes());
        }
        let gx = ((b32x + 1) / 2).max(1) as u32;
        let gy = ((b32y + 1) / 2).max(1) as u32;
        pack_gx = pack_gx.max(gx);
        pack_gy = pack_gy.max(gy);
    }
    (quant, pack_jobs, pack_gx, pack_gy, count8, count32)
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
    let mut quant_res = (1i32 << bits.min(30)) as f32;
    if quant_res > 4096.0 {
        quant_res = 4096.0;
    }
    let code = quant_code(level, component, band);
    let decoded = decode_quant_code(code as u8);
    if decoded == 0.0 {
        quant_res
    } else {
        1.0 / decoded
    }
}

fn quant_code(level: i32, component: i32, band: i32) -> u32 {
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
    let mut quant_res = (1i32 << bits.min(30)) as f32;
    if quant_res > 4096.0 {
        quant_res = 4096.0;
    }
    let v = (1.0f32 / quant_res).to_bits();
    let mut e = ((v >> 23) & 0xff) as i32 - 127 - 4;
    let m = ((v >> 20) & 7) as i32;
    e = -e;
    if e < 0 {
        e = 0;
    }
    if e > 20 {
        e = 20;
    }
    ((e << 3) | m) as u32
}

fn decode_quant_code(quant_code: u8) -> f32 {
    let e = 4 - (quant_code >> 3) as i32;
    let m = (quant_code & 7) as i32;
    let shift = 20 + e;
    let mut mag = (8 + m) as f32;
    if shift >= 0 {
        mag *= (1i32 << shift.min(30)) as f32;
    } else {
        mag /= (1i32 << (-shift).min(30)) as f32;
    }
    mag / (8.0 * 1024.0 * 1024.0)
}
