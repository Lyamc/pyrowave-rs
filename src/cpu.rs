//! CPU backend. No device is created. The wavelet and the per-block reduction
//! both go through `simd_dwt`.

use crate::packet::{block_record, FrameBits};
use crate::simd_dwt::{forward_wavelet, quantize_block, reduce_max_abs};
use crate::Chroma;

pub fn encode_cpu(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: u32,
    height: u32,
    chroma: Chroma,
) -> Result<FrameBits, String> {
    if width < 2 || height < 2 || width % 2 != 0 || height % 2 != 0 {
        return Err("width and height must be even and at least 2".into());
    }
    let chroma_444 = chroma == Chroma::Yuv444;
    let (cw, ch) = if chroma_444 {
        (width as usize, height as usize)
    } else {
        (width as usize / 2, height as usize / 2)
    };
    if y.len() < width as usize * height as usize {
        return Err("Y plane is short".into());
    }
    if u.len() < cw * ch || v.len() < cw * ch {
        return Err("chroma plane is short".into());
    }

    let mut planes = [
        to_f32(y, width as usize * height as usize),
        to_f32(u, cw * ch),
        to_f32(v, cw * ch),
    ];
    let dims = [
        (width as usize, height as usize),
        (cw, ch),
        (cw, ch),
    ];
    for (plane, (w, h)) in planes.iter_mut().zip(dims) {
        forward_wavelet(plane, w, h);
    }

    let mut blocks = Vec::new();
    let mut samples = vec![0.0f32; 32 * 32];
    let mut quant = vec![0i16; 32 * 32];
    let mut index = 0u32;
    for (plane, (w, h)) in planes.iter().zip(dims) {
        let mut y0 = 0;
        while y0 < h {
            let bh = (h - y0).min(32);
            let mut x0 = 0;
            while x0 < w {
                let bw = (w - x0).min(32);
                let n = bw * bh;
                let mut at = 0;
                for row in 0..bh {
                    let start = (y0 + row) * w + x0;
                    samples[at..at + bw].copy_from_slice(&plane[start..start + bw]);
                    at += bw;
                }
                let energy = reduce_max_abs(&samples[..n]);
                quantize_block(&samples[..n], &mut quant[..n]);
                // Always keep the first luma block so a flat frame still has
                // a packet. Later blocks ride on the SIMD energy.
                if index == 0 || energy > 0.0 {
                    let mut payload = vec![0u8; n * 2 + (n * 2) % 4];
                    for (i, s) in quant[..n].iter().enumerate() {
                        payload[i * 2..i * 2 + 2].copy_from_slice(&s.to_le_bytes());
                    }
                    blocks.push(block_record(index, &payload));
                }
                index += 1;
                x0 += 32;
            }
            y0 += 32;
        }
    }
    if blocks.is_empty() {
        return Err("cpu encode produced no blocks".into());
    }
    Ok(FrameBits {
        width,
        height,
        chroma_444,
        blocks: blocks
            .into_iter()
            .map(|bytes| crate::packet::BlockMeta { bytes })
            .collect(),
    })
}

fn to_f32(src: &[u8], n: usize) -> Vec<f32> {
    src[..n].iter().map(|p| f32::from(*p) - 128.0).collect()
}
