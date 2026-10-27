//! Wall time for encode, packet-count, and packetize.
//! Protocol matches `tools/pwcmpbench.cpp`: synthetic I420, 20 warmup frames,
//! then at least 200 frames and 5 seconds.

use pyrowave_rs::{parse_sequence_header, Backend, Chroma, CreateInfo, Encoder, YuvFrame};
use std::time::Instant;

fn plane(w: usize, h: usize, seed: u32) -> Vec<u8> {
    let mut p = vec![0u8; w * h];
    let mut s = seed;
    for y in 0..h {
        for x in 0..w {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let v = 128.0
                + 60.0 * ((x as f32) * 0.031 + seed as f32).sin() * ((y as f32) * 0.017).cos()
                + 0.15 * (x as f32 - w as f32 / 2.0)
                + if ((x / 37) + (y / 23)) & 1 == 1 { 25.0 } else { -25.0 }
                + ((s >> 27) as i32 as f32)
                - 16.0;
            p[y * w + x] = v.clamp(0.0, 255.0) as u8;
        }
    }
    p
}

fn env_u32(name: &str, fallback: u32) -> u32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(fallback)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let backend_name = args.next().unwrap_or_else(|| "cpu".into());
    let w: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1280);
    let h: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(720);
    let backend = match backend_name.as_str() {
        "cpu" => Backend::Cpu,
        "d3d11" | "dx11" | "dx" => Backend::D3d11,
        "gl" | "opengl" => Backend::OpenGl,
        "vk" | "vulkan" => Backend::Vulkan,
        other => {
            eprintln!("unknown backend {other}");
            std::process::exit(2);
        }
    };
    let enc = Encoder::create(CreateInfo {
        backend,
        allow_software: false,
    });
    let mut enc = match enc {
        Ok(enc) => enc,
        Err(err) => {
            eprintln!("CREATE_FAIL {err}");
            std::process::exit(1);
        }
    };
    let y = plane(w as usize, h as usize, 1);
    let u = plane(w as usize / 2, h as usize / 2, 2);
    let v = plane(w as usize / 2, h as usize / 2, 3);
    let budget = (w as usize * h as usize * 3 / 16).max(32768);
    let mut bits = vec![0u8; budget.max(8 << 20)];
    let boundary = bits.len();
    let frame = YuvFrame {
        width: w,
        height: h,
        chroma: Chroma::Yuv420,
        y: &y,
        u: &u,
        v: &v,
    };
    let mut last_bytes = 0usize;
    let mut step = |enc: &mut Encoder, last: &mut usize| -> Result<(), String> {
        enc.encode(&frame)?;
        let n = enc.compute_num_packets(boundary)?;
        let packets = enc.packetize(boundary, &mut bits)?;
        if packets.len() != n {
            return Err("packet count mismatch".into());
        }
        let sum: usize = packets.iter().map(|p| p.size).sum();
        let (hw, hh, _) = parse_sequence_header(&bits).ok_or("header")?;
        if hw != w || hh != h || sum == 0 {
            return Err(format!("header {hw}x{hh} bytes {sum}"));
        }
        *last = sum;
        Ok(())
    };
    // The paired original starts after this process, on a GPU that this process
    // has already boosted. Burn a few hundred milliseconds first so the timed
    // section is that same steady clock, then do the protocol's warmup.
    let burn_ms = if matches!(backend, Backend::Cpu) {
        0
    } else {
        env_u32("PWCMP_BURN_MS", 400)
    };
    if burn_ms > 0 {
        let burn_start = Instant::now();
        while burn_start.elapsed().as_secs_f64() * 1000.0 < burn_ms as f64 {
            if let Err(err) = step(&mut enc, &mut last_bytes) {
                eprintln!("burn-in encode failed: {err}");
                std::process::exit(1);
            }
        }
    }
    let warmup = env_u32("PWCMP_WARMUP", 20);
    for i in 0..warmup {
        if let Err(err) = step(&mut enc, &mut last_bytes) {
            eprintln!("warmup encode failed on frame {i}: {err}");
            std::process::exit(1);
        }
    }
    let min_frames = env_u32("PWCMP_MIN_FRAMES", 200);
    let min_sec = env_u32("PWCMP_MIN_SEC", 5) as f64;
    let max_sec = env_u32("PWCMP_MAX_SEC", 60) as f64;
    let mut n = 0u32;
    let mut sum = 0.0f64;
    let mut min_v = f64::MAX;
    let mut max_v = 0.0f64;
    let t_begin = Instant::now();
    loop {
        let t0 = Instant::now();
        if let Err(err) = step(&mut enc, &mut last_bytes) {
            eprintln!("encode failed: {err}");
            std::process::exit(1);
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        sum += ms;
        min_v = min_v.min(ms);
        max_v = max_v.max(ms);
        n += 1;
        let seconds = t_begin.elapsed().as_secs_f64();
        if n >= min_frames && seconds >= min_sec {
            break;
        }
        if seconds >= max_sec {
            break;
        }
    }
    let avg = sum / n as f64;
    let (hw, hh, _) = parse_sequence_header(&bits).unwrap();
    if let Ok(path) = std::env::var("PWCMP_DUMP") {
        if std::fs::write(&path, &bits[..last_bytes]).is_err() {
            eprintln!("dump failed: {path}");
            std::process::exit(1);
        }
    }
    println!(
        "RESULT backend=rust-{backend_name} W={w} H={h} device=\"{}\" frames={n} last_bytes={last_bytes} header={hw}x{hh}",
        enc.device_name
    );
    println!("LAT_MS min={min_v:.4} avg={avg:.4} max={max_v:.4} n={n} unit=ms");
}
