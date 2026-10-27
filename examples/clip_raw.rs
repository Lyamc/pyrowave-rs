//! Encode a raw 8-bit 4:2:0 file as a PWV1 sequence.
//!
//! `clip_raw <cpu|d3d11|opengl|vulkan> <i420> <width> <height> <frames> <fps_num> <fps_den> <out.raw>`
//!
//! The file is the reference layout: one 24-byte `PWV1Header`, then each frame
//! as a little-endian byte count and one packetized frame.

use pyrowave_rs::{pwv1_header, Backend, Chroma, CreateInfo, Encoder, YuvFrame};
use std::env;
use std::fs::File;
use std::io::{Read, Write};
use std::process::ExitCode;

#[used]
#[no_mangle]
pub static NvOptimusEnablement: u32 = 1;

#[used]
#[no_mangle]
pub static AmdPowerXpressRequestHighPerformance: i32 = 1;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let Some(backend) = args.next() else {
        eprintln!("usage: clip_raw <backend> <i420> <w> <h> <frames> <fps_num> <fps_den> <out>");
        return ExitCode::from(2);
    };
    let Some(path) = args.next() else {
        eprintln!("missing i420 path");
        return ExitCode::from(2);
    };
    let Some(width) = args.next().and_then(|s| s.parse::<u32>().ok()) else {
        eprintln!("missing width");
        return ExitCode::from(2);
    };
    let Some(height) = args.next().and_then(|s| s.parse::<u32>().ok()) else {
        eprintln!("missing height");
        return ExitCode::from(2);
    };
    let Some(frames) = args.next().and_then(|s| s.parse::<u32>().ok()) else {
        eprintln!("missing frame count");
        return ExitCode::from(2);
    };
    let Some(fps_num) = args.next().and_then(|s| s.parse::<u32>().ok()) else {
        eprintln!("missing fps numerator");
        return ExitCode::from(2);
    };
    let Some(fps_den) = args.next().and_then(|s| s.parse::<u32>().ok()) else {
        eprintln!("missing fps denominator");
        return ExitCode::from(2);
    };
    let Some(out_path) = args.next() else {
        eprintln!("missing output path");
        return ExitCode::from(2);
    };
    let backend = match backend.as_str() {
        "cpu" => Backend::Cpu,
        "d3d11" | "dx" => Backend::D3d11,
        "opengl" | "gl" => Backend::OpenGl,
        "vulkan" | "vk" => Backend::Vulkan,
        other => {
            eprintln!("unknown backend {other}");
            return ExitCode::from(2);
        }
    };
    if width == 0 || height == 0 || width % 2 != 0 || height % 2 != 0 || frames == 0 {
        eprintln!("frame size must be even and non-zero");
        return ExitCode::from(2);
    }

    match run(backend, &path, width, height, frames, fps_num, fps_den, &out_path) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("ERROR {err}");
            ExitCode::from(1)
        }
    }
}

fn run(
    backend: Backend,
    path: &str,
    width: u32,
    height: u32,
    frames: u32,
    fps_num: u32,
    fps_den: u32,
    out_path: &str,
) -> Result<(), String> {
    // AN515 has no hardware Vulkan ICD. Lavapipe is that machine's Vulkan
    // device. A hardware device still wins when one is present (vendor 0x10DE
    // replaces anything else). D3D11 and OpenGL stay hardware-only.
    let allow_software = matches!(backend, Backend::Vulkan);
    let mut enc = Encoder::create(CreateInfo { backend, allow_software })?;
    eprintln!(
        "PATH backend={backend:?} gpu={} device={}",
        enc.uses_gpu(),
        enc.device_name
    );
    let y_len = (width as usize).saturating_mul(height as usize);
    let c_len = y_len / 4;
    let frame_bytes = y_len + c_len + c_len;
    let mut file = File::open(path).map_err(|err| format!("open {path}: {err}"))?;
    let mut raw = vec![0u8; frame_bytes];
    let mut out = File::create(out_path).map_err(|err| format!("create {out_path}: {err}"))?;
    out.write_all(&pwv1_header(width, height, fps_num, fps_den))
        .map_err(|err| format!("write header: {err}"))?;
    let target = Encoder::default_target(width, height);
    let mut bits = vec![0u8; target as usize];
    for index in 0..frames {
        file.read_exact(&mut raw)
            .map_err(|err| format!("read frame {index}: {err}"))?;
        let (y, rest) = raw.split_at(y_len);
        let (u, v) = rest.split_at(c_len);
        enc.encode_target(
            &YuvFrame { width, height, chroma: Chroma::Yuv420, y, u, v },
            target,
        )?;
        let packets = enc.packetize(bits.len(), &mut bits)?;
        if packets.len() != 1 {
            return Err(format!("frame {index} split into {} packets", packets.len()));
        }
        let size = packets[0].size;
        out.write_all(&(size as u32).to_le_bytes())
            .map_err(|err| format!("write size: {err}"))?;
        out.write_all(&bits[..size])
            .map_err(|err| format!("write frame: {err}"))?;
        eprintln!("FRAME {index} bytes={size} hash={:016x}", fnv(&bits[..size]));
    }
    out.flush().map_err(|err| format!("flush: {err}"))?;
    eprintln!("WROTE {out_path} frames={frames} target={target}");
    Ok(())
}

fn fnv(data: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for byte in data {
        h ^= u64::from(*byte);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}
