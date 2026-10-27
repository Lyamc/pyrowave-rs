//! Band list shared by the CPU, D3D11, and OpenGL backends. Dimensions follow
//! the aligned pyramid in the D3D11 encoder: width and height round up to 32
//! and then to at least 128, and each level halves them.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BandJob {
    pub component: u8,
    pub level: u8,
    pub width: u32,
    pub height: u32,
}

fn align_frame(v: u32) -> u32 {
    let a = (v + 31) & !31;
    if a < 128 {
        128
    } else {
        a
    }
}

/// One job per component per level that still has samples. 4:2:0 chroma
/// starts at level 1 with half the luma resolution, matching the ports.
pub fn band_jobs(width: u32, height: u32, chroma_444: bool) -> Vec<BandJob> {
    let aw = align_frame(width);
    let ah = align_frame(height);
    let mut jobs = Vec::new();
    let components = if chroma_444 { 3 } else { 1 };
    for level in 0..5 {
        let w = (aw >> level).max(1);
        let h = (ah >> level).max(1);
        for c in 0..components {
            jobs.push(BandJob {
                component: c,
                level: level as u8,
                width: w,
                height: h,
            });
        }
        if !chroma_444 && level >= 1 {
            let cw = (aw >> (level + 1)).max(1);
            let ch = (ah >> (level + 1)).max(1);
            for c in 1..3 {
                jobs.push(BandJob {
                    component: c,
                    level: level as u8,
                    width: cw,
                    height: ch,
                });
            }
        }
    }
    jobs
}

/// Frequency-band job list used by the quant and pack dispatches.
/// Level 4 codes subband 0; every level codes subbands 1..3.
/// `gl/src/encoder.cpp` `build_jobs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CodingJob {
    pub level: u8,
    pub component: u8,
    pub band: u8,
    pub width: i32,
    pub height: i32,
    pub pitch: i32,
    pub off8: i32,
    pub stride8: i32,
    pub off32: i32,
    pub stride32: i32,
}

#[derive(Clone, Debug)]
pub struct CodingLayout {
    pub aligned_w: i32,
    pub aligned_h: i32,
    pub jobs: Vec<CodingJob>,
    pub total8: i32,
    pub total32: i32,
}

pub fn coding_layout(width: u32, height: u32, chroma_444: bool) -> CodingLayout {
    let aw = align_frame(width) as i32;
    let ah = align_frame(height) as i32;
    let mut jobs = Vec::new();
    let mut count8 = 0i32;
    let mut count32 = 0i32;
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
            let first = if level == 4 { 0 } else { 1 };
            for band in first..4 {
                jobs.push(CodingJob {
                    level: level as u8,
                    component: component as u8,
                    band: band as u8,
                    width: w,
                    height: h,
                    pitch,
                    off8: count8,
                    stride8: b8x,
                    off32: count32,
                    stride32: b32x,
                });
                count8 += b8x * b8y;
                count32 += b32x * ((h + 31) / 32);
            }
        }
    }
    CodingLayout {
        aligned_w: aw,
        aligned_h: ah,
        jobs,
        total8: count8,
        total32: count32,
    }
}
