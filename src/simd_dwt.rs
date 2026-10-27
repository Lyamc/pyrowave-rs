//! CDF 9/7 lifting. Horizontal groups of rows and vertical groups of columns
//! run as `std::simd` lanes. Constants match `src/shaders/dwt.hlsl`.

use std::simd::num::SimdFloat;
use std::simd::{Simd, StdFloat};

pub const ALPHA: f32 = -1.586134342059924;
pub const BETA: f32 = -0.052980118572961;
pub const GAMMA: f32 = 0.882911075530934;
pub const DELTA: f32 = 0.443506852043971;
pub const K: f32 = 1.230174104914001;
pub const INV_K: f32 = 1.0 / K;

/// In-place one-level 2D forward CDF 9/7. `plane` is row-major `width * height`.
pub fn forward_wavelet(plane: &mut [f32], width: usize, height: usize) {
    assert!(width >= 2 && height >= 2, "wavelet plane is at least 2x2");
    assert_eq!(plane.len(), width * height, "plane length");
    lift_horizontal(plane, width, height);
    lift_vertical(plane, width, height);
}

/// Max |sample| of one block. The reduction itself is `f32x8`.
pub fn reduce_max_abs(samples: &[f32]) -> f32 {
    let mut acc = Simd::<f32, 8>::splat(0.0);
    let mut i = 0;
    while i + 8 <= samples.len() {
        let v = Simd::<f32, 8>::from_slice(&samples[i..i + 8]);
        acc = acc.simd_max(v.abs());
        i += 8;
    }
    let mut max_abs = acc.reduce_max();
    while i < samples.len() {
        max_abs = max_abs.max(samples[i].abs());
        i += 1;
    }
    max_abs
}

/// Quantize a block to i16 with an f32x8 scale. Quant 0 keeps the coefficient
/// magnitude; the scale only moves it onto an integer grid.
pub fn quantize_block(samples: &[f32], out: &mut [i16]) {
    assert_eq!(samples.len(), out.len());
    let scale = Simd::<f32, 8>::splat(4.0);
    let mut i = 0;
    while i + 8 <= samples.len() {
        let v = Simd::<f32, 8>::from_slice(&samples[i..i + 8]) * scale;
        let rounded = v.round();
        let arr = rounded.to_array();
        for lane in 0..8 {
            out[i + lane] = arr[lane].clamp(-32767.0, 32767.0) as i16;
        }
        i += 8;
    }
    while i < samples.len() {
        out[i] = (samples[i] * 4.0).round().clamp(-32767.0, 32767.0) as i16;
        i += 1;
    }
}

fn lift_horizontal(plane: &mut [f32], width: usize, height: usize) {
    let mut y = 0;
    while y + 8 <= height {
        lift_rows8(plane, width, y);
        y += 8;
    }
    while y < height {
        lift_row1(plane, width, y);
        y += 1;
    }
}

fn lift_vertical(plane: &mut [f32], width: usize, height: usize) {
    let mut x = 0;
    while x + 8 <= width {
        lift_cols8(plane, width, height, x);
        x += 8;
    }
    while x < width {
        lift_col1(plane, width, height, x);
        x += 1;
    }
}

fn load8(plane: &[f32], width: usize, y: usize, x: usize) -> Simd<f32, 8> {
    let mut a = [0.0f32; 8];
    let mut i = 0;
    while i < 8 {
        a[i] = plane[(y + i) * width + x];
        i += 1;
    }
    Simd::from_array(a)
}

fn store8(plane: &mut [f32], width: usize, y: usize, x: usize, v: Simd<f32, 8>) {
    let a = v.to_array();
    let mut i = 0;
    while i < 8 {
        plane[(y + i) * width + x] = a[i];
        i += 1;
    }
}

fn lift_rows8(plane: &mut [f32], width: usize, y: usize) {
    let alpha = Simd::<f32, 8>::splat(ALPHA);
    let beta = Simd::<f32, 8>::splat(BETA);
    let gamma = Simd::<f32, 8>::splat(GAMMA);
    let delta = Simd::<f32, 8>::splat(DELTA);
    let k = Simd::<f32, 8>::splat(K);
    let inv_k = Simd::<f32, 8>::splat(INV_K);
    let step = |plane: &mut [f32], start: usize, coeff: Simd<f32, 8>| {
        let mut i = start;
        while i + 1 < width {
            let left = load8(plane, width, y, i - 1);
            let mid = load8(plane, width, y, i);
            let right = load8(plane, width, y, i + 1);
            store8(plane, width, y, i, mid + coeff * (left + right));
            i += 2;
        }
    };
    step(plane, 1, alpha);
    if width % 2 == 0 && width >= 2 {
        let i = width - 1;
        let left = load8(plane, width, y, i - 1);
        let mid = load8(plane, width, y, i);
        store8(plane, width, y, i, mid + alpha * (left + left));
    }
    step(plane, 2, beta);
    if width % 2 == 1 && width >= 3 {
        let i = width - 1;
        let left = load8(plane, width, y, i - 1);
        let mid = load8(plane, width, y, i);
        store8(plane, width, y, i, mid + beta * (left + left));
    }
    step(plane, 3, gamma);
    if width % 2 == 0 && width >= 4 {
        let i = width - 1;
        let left = load8(plane, width, y, i - 1);
        let mid = load8(plane, width, y, i);
        store8(plane, width, y, i, mid + gamma * (left + left));
    }
    step(plane, 4, delta);
    if width >= 5 && width % 2 == 1 {
        let i = width - 1;
        let left = load8(plane, width, y, i - 1);
        let mid = load8(plane, width, y, i);
        store8(plane, width, y, i, mid + delta * (left + left));
    }
    let mut i = 0;
    while i < width {
        store8(plane, width, y, i, load8(plane, width, y, i) * inv_k);
        if i + 1 < width {
            store8(plane, width, y, i + 1, load8(plane, width, y, i + 1) * k);
        }
        i += 2;
    }
}

fn lift_cols8(plane: &mut [f32], width: usize, height: usize, x: usize) {
    let alpha = Simd::<f32, 8>::splat(ALPHA);
    let beta = Simd::<f32, 8>::splat(BETA);
    let gamma = Simd::<f32, 8>::splat(GAMMA);
    let delta = Simd::<f32, 8>::splat(DELTA);
    let k = Simd::<f32, 8>::splat(K);
    let inv_k = Simd::<f32, 8>::splat(INV_K);
    let at = |y: usize| y * width + x;
    let step = |plane: &mut [f32], start: usize, coeff: Simd<f32, 8>| {
        let mut i = start;
        while i + 1 < height {
            let left = Simd::<f32, 8>::from_slice(&plane[at(i - 1)..at(i - 1) + 8]);
            let mid = Simd::<f32, 8>::from_slice(&plane[at(i)..at(i) + 8]);
            let right = Simd::<f32, 8>::from_slice(&plane[at(i + 1)..at(i + 1) + 8]);
            (mid + coeff * (left + right)).copy_to_slice(&mut plane[at(i)..at(i) + 8]);
            i += 2;
        }
    };
    step(plane, 1, alpha);
    if height % 2 == 0 && height >= 2 {
        let i = height - 1;
        let left = Simd::<f32, 8>::from_slice(&plane[at(i - 1)..at(i - 1) + 8]);
        let mid = Simd::<f32, 8>::from_slice(&plane[at(i)..at(i) + 8]);
        (mid + alpha * (left + left)).copy_to_slice(&mut plane[at(i)..at(i) + 8]);
    }
    step(plane, 2, beta);
    if height % 2 == 1 && height >= 3 {
        let i = height - 1;
        let left = Simd::<f32, 8>::from_slice(&plane[at(i - 1)..at(i - 1) + 8]);
        let mid = Simd::<f32, 8>::from_slice(&plane[at(i)..at(i) + 8]);
        (mid + beta * (left + left)).copy_to_slice(&mut plane[at(i)..at(i) + 8]);
    }
    step(plane, 3, gamma);
    if height % 2 == 0 && height >= 4 {
        let i = height - 1;
        let left = Simd::<f32, 8>::from_slice(&plane[at(i - 1)..at(i - 1) + 8]);
        let mid = Simd::<f32, 8>::from_slice(&plane[at(i)..at(i) + 8]);
        (mid + gamma * (left + left)).copy_to_slice(&mut plane[at(i)..at(i) + 8]);
    }
    step(plane, 4, delta);
    if height >= 5 && height % 2 == 1 {
        let i = height - 1;
        let left = Simd::<f32, 8>::from_slice(&plane[at(i - 1)..at(i - 1) + 8]);
        let mid = Simd::<f32, 8>::from_slice(&plane[at(i)..at(i) + 8]);
        (mid + delta * (left + left)).copy_to_slice(&mut plane[at(i)..at(i) + 8]);
    }
    let mut i = 0;
    while i < height {
        let row = at(i);
        (Simd::<f32, 8>::from_slice(&plane[row..row + 8]) * inv_k).copy_to_slice(&mut plane[row..row + 8]);
        if i + 1 < height {
            let row = at(i + 1);
            (Simd::<f32, 8>::from_slice(&plane[row..row + 8]) * k).copy_to_slice(&mut plane[row..row + 8]);
        }
        i += 2;
    }
}

fn lift_row1(plane: &mut [f32], width: usize, y: usize) {
    let row = y * width;
    let mut tmp = plane[row..row + width].to_vec();
    lift_scalar(&mut tmp);
    plane[row..row + width].copy_from_slice(&tmp);
}

fn lift_col1(plane: &mut [f32], width: usize, height: usize, x: usize) {
    let mut tmp = vec![0.0f32; height];
    for y in 0..height {
        tmp[y] = plane[y * width + x];
    }
    lift_scalar(&mut tmp);
    for y in 0..height {
        plane[y * width + x] = tmp[y];
    }
}

fn lift_scalar(v: &mut [f32]) {
    let n = v.len();
    if n < 2 {
        return;
    }
    let mut i = 1;
    while i + 1 < n {
        v[i] += ALPHA * (v[i - 1] + v[i + 1]);
        i += 2;
    }
    if n % 2 == 0 {
        let i = n - 1;
        v[i] += ALPHA * (v[i - 1] + v[i - 1]);
    }
    let mut i = 2;
    while i + 1 < n {
        v[i] += BETA * (v[i - 1] + v[i + 1]);
        i += 2;
    }
    if n % 2 == 1 && n >= 3 {
        let i = n - 1;
        v[i] += BETA * (v[i - 1] + v[i - 1]);
    }
    let mut i = 3;
    while i + 1 < n {
        v[i] += GAMMA * (v[i - 1] + v[i + 1]);
        i += 2;
    }
    if n % 2 == 0 && n >= 4 {
        let i = n - 1;
        v[i] += GAMMA * (v[i - 1] + v[i - 1]);
    }
    let mut i = 4;
    while i + 1 < n {
        v[i] += DELTA * (v[i - 1] + v[i + 1]);
        i += 2;
    }
    if n >= 5 && n % 2 == 1 {
        let i = n - 1;
        v[i] += DELTA * (v[i - 1] + v[i - 1]);
    }
    let mut i = 0;
    while i < n {
        v[i] *= INV_K;
        if i + 1 < n {
            v[i + 1] *= K;
        }
        i += 2;
    }
}
