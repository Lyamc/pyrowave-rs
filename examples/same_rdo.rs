//! Same-session packet compare. Every Rust backend and, on Windows, the
//! original Vulkan encoder (`libpyrowave-shared-1.dll`) packetize one
//! synthetic 4:2:0 frame. The frame generator matches `tools/pwcmpbench.cpp`
//! `make_plane`. Don't-care bits follow `docs/SPEC.md` §5.

use pyrowave_rs::{parse_sequence_header, Backend, Chroma, CreateInfo, Encoder, YuvFrame};
use std::env;

#[used]
#[no_mangle]
pub static NvOptimusEnablement: u32 = 1;

#[used]
#[no_mangle]
pub static AmdPowerXpressRequestHighPerformance: i32 = 1;

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

fn fnv(data: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for byte in data {
        h ^= u64::from(*byte);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// End of the defined sign payload, and how many low bits of the last sign
/// byte are defined. Stage 5 §4: planes then signs; padding and the high bits
/// of the last sign byte are don't-care.
fn sign_end(block: &[u8]) -> Option<(usize, u32)> {
    if block.len() < 8 {
        return None;
    }
    let word0 = u32::from_le_bytes(block[0..4].try_into().ok()?);
    let words = ((word0 >> 16) & 0xfff) as usize;
    if words < 2 || words * 4 != block.len() {
        return None;
    }
    let ballot = word0 & 0xffff;
    let n = ballot.count_ones() as usize;
    let plane_at = 8 + 3 * n;
    if plane_at > block.len() {
        return None;
    }
    let mut cursor = plane_at;
    let mut sign_bits = 0u32;
    let mut slot = 0usize;
    for sub in 0..16 {
        if ballot & (1 << sub) == 0 {
            continue;
        }
        let code_at = 8 + 2 * slot;
        let q_at = 8 + 2 * n + slot;
        if q_at >= block.len() || code_at + 2 > block.len() {
            return None;
        }
        let code = u16::from_le_bytes(block[code_at..code_at + 2].try_into().ok()?);
        let q_bits = u32::from(block[q_at] & 0xf);
        for sb in 0..8 {
            let out_planes = ((u32::from(code) >> (2 * sb)) & 3) + q_bits;
            if out_planes == 0 {
                continue;
            }
            let end = cursor + out_planes as usize;
            if end > block.len() {
                return None;
            }
            let mut significant = 0u32;
            for byte in &block[cursor..end] {
                significant |= u32::from(*byte);
            }
            sign_bits += significant.count_ones();
            cursor = end;
        }
        slot += 1;
    }
    if slot != n {
        return None;
    }
    let sign_bytes = ((sign_bits + 7) / 8) as usize;
    let defined = if sign_bits == 0 { 8 } else { ((sign_bits - 1) % 8) + 1 };
    Some((cursor + sign_bytes, defined))
}

fn mask_block(block: &[u8]) -> Vec<u8> {
    let mut out = block.to_vec();
    if let Some((end, defined_bits)) = sign_end(block) {
        if end > 0 && end <= out.len() && defined_bits < 8 {
            let last = end - 1;
            let keep = (1u16 << defined_bits) as u8 - 1;
            out[last] &= keep;
        }
        if end < out.len() {
            for byte in &mut out[end..] {
                *byte = 0;
            }
        }
    }
    out
}

fn split_blocks(bits: &[u8]) -> Result<Vec<&[u8]>, String> {
    if bits.len() < 8 {
        return Err(format!("record is {} bytes", bits.len()));
    }
    let mut i = 8usize;
    let mut blocks = Vec::new();
    while i < bits.len() {
        if i + 4 > bits.len() {
            return Err(format!("trailing {} bytes after block {}", bits.len() - i, blocks.len()));
        }
        let word0 = u32::from_le_bytes(bits[i..i + 4].try_into().unwrap());
        let words = ((word0 >> 16) & 0xfff) as usize;
        if words < 2 {
            return Err(format!("block {} payload_words={words}", blocks.len()));
        }
        let n = words * 4;
        if i + n > bits.len() {
            return Err(format!(
                "block {} needs {n} bytes, {} remain",
                blocks.len(),
                bits.len() - i
            ));
        }
        blocks.push(&bits[i..i + n]);
        i += n;
    }
    Ok(blocks)
}

fn compare(tag: &str, got: &[u8], oracle: &[u8]) -> String {
    let Some((gw, gh, gc)) = parse_sequence_header(got) else {
        return "cmp=header-error".into();
    };
    let Some((ow, oh, oc)) = parse_sequence_header(oracle) else {
        return "cmp=oracle-header-error".into();
    };
    if (gw, gh, gc) != (ow, oh, oc) {
        return format!("cmp=header {gw}x{gh} c={gc} vs {ow}x{oh} c={oc}");
    }
    let gb = match split_blocks(got) {
        Ok(v) => v,
        Err(err) => return format!("cmp=split {err}"),
    };
    let ob = match split_blocks(oracle) {
        Ok(v) => v,
        Err(err) => return format!("cmp=oracle-split {err}"),
    };
    if gb.len() != ob.len() {
        return format!("cmp=block-count {} vs {}", gb.len(), ob.len());
    }
    let mut raw_diff = 0usize;
    let mut masked_diff = 0usize;
    let mut first = String::new();
    for (i, (g, o)) in gb.iter().zip(ob.iter()).enumerate() {
        if g == o {
            continue;
        }
        raw_diff += 1;
        let gm = mask_block(g);
        let om = mask_block(o);
        if gm != om {
            masked_diff += 1;
            if first.is_empty() {
                let gw0 = u32::from_le_bytes(g[0..4].try_into().unwrap());
                let ow0 = u32::from_le_bytes(o[0..4].try_into().unwrap());
                let gw1 = u32::from_le_bytes(g[4..8].try_into().unwrap());
                let ow1 = u32::from_le_bytes(o[4..8].try_into().unwrap());
                let gh: String = g.iter().take(32).map(|b| format!("{b:02x}")).collect();
                let oh: String = o.iter().take(32).map(|b| format!("{b:02x}")).collect();
                first = format!(
                    " first={i} word0={gw0:#x}/{ow0:#x} word1={gw1:#x}/{ow1:#x} len={}/{} g={gh} o={oh}",
                    g.len(),
                    o.len()
                );
            }
        }
    }
    let mut hi_n = 0usize;
    let mut hi_eq = 0usize;
    let mut hi_first = String::new();
    for (g, o) in gb.iter().zip(ob.iter()) {
        if g.len() < 8 || o.len() < 8 {
            continue;
        }
        let idx = u32::from_le_bytes(g[4..8].try_into().unwrap()) >> 8;
        if idx < 700 {
            continue;
        }
        hi_n += 1;
        if g == o {
            hi_eq += 1;
        } else if hi_first.is_empty() {
            let mut at = 0usize;
            let n = g.len().min(o.len());
            while at < n && g[at] == o[at] {
                at += 1;
            }
            hi_first = format!(" hi_idx={idx} diverge_at={at} glen={} olen={}", g.len(), o.len());
        }
    }
    let hi = format!(" hi_blocks={hi_n} hi_equal={hi_eq}{hi_first}");
    if masked_diff == 0 {
        if raw_diff == 0 {
            format!("cmp=equal blocks={}{hi}", gb.len())
        } else {
            format!("cmp=equal-masked blocks={} dontcare={raw_diff}{hi}", gb.len())
        }
    } else {
        format!("cmp=FAIL blocks={} masked_diff={masked_diff} raw_diff={raw_diff}{first}{hi} tag={tag}", gb.len())
    }
}

fn record(backend: &str, w: u32, h: u32, target: u32, bits: &[u8], cmp: &str) -> usize {
    let (hw, hh, chroma) = parse_sequence_header(bits).unwrap_or((0, 0, false));
    let gb = split_blocks(bits).unwrap_or_default();
    let mut masked = bits[..8].to_vec();
    for block in &gb {
        masked.extend(mask_block(block));
    }
    println!(
        "RECORD backend={backend} W={w} H={h} target={target} bytes={} header={hw}x{hh} chroma444={} blocks={} walked={} hash={:016x} {cmp}",
        bits.len(),
        u8::from(chroma),
        gb.len(),
        bits.len(),
        fnv(&masked),
    );
    bits.len()
}

fn encode_rust(backend: Backend, y: &[u8], u: &[u8], v: &[u8], w: u32, h: u32, target: u32) -> Result<Vec<u8>, String> {
    let mut enc = Encoder::create(CreateInfo { backend, allow_software: false })?;
    println!("PATH backend={backend:?} gpu={} device={}", enc.uses_gpu(), enc.device_name);
    enc.encode_target(
        &YuvFrame { width: w, height: h, chroma: Chroma::Yuv420, y, u, v },
        target,
    )?;
    let boundary = (16usize << 20).max(target as usize);
    let n = enc.compute_num_packets(boundary)?;
    let mut bits = vec![0u8; boundary];
    let packets = enc.packetize(boundary, &mut bits)?;
    if packets.len() != n {
        return Err(format!("packet count {n} != {}", packets.len()));
    }
    let mut walked = 0usize;
    for packet in packets.iter() {
        if packet.offset != walked {
            return Err(format!("packet gap at {walked}"));
        }
        walked += packet.size;
    }
    bits.truncate(walked);
    let Some((hw, hh, chroma)) = parse_sequence_header(&bits) else {
        return Err("sequence header did not parse".into());
    };
    if (hw, hh, chroma) != (w, h, false) {
        return Err(format!("header {hw}x{hh} chroma={chroma}"));
    }
    Ok(bits)
}

#[cfg(windows)]
mod original {
    use std::ffi::c_void;
    use std::os::raw::c_char;

    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryA(name: *const u8) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
        fn SetDllDirectoryA(path: *const u8) -> i32;
        fn GetLastError() -> u32;
    }

    #[repr(C)]
    struct EncInfo {
        device: *mut c_void,
        width: i32,
        height: i32,
        chroma: i32,
    }
    #[repr(C)]
    struct CpuBuf {
        data: [*mut c_void; 3],
        row_stride: [usize; 3],
        plane_size: [usize; 3],
        width: i32,
        height: i32,
        format: i32,
    }
    #[repr(C)]
    struct Rate {
        maximum_bitstream_size: usize,
    }
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Pkt {
        offset: usize,
        size: usize,
    }

    type FnDefault = unsafe extern "C" fn(*mut *mut c_void) -> i32;
    type FnCompat = unsafe extern "C" fn(u32, u32, *const c_void, *const c_void, *const c_void, i32, *mut *mut c_void) -> i32;
    type FnEncCreate = unsafe extern "C" fn(*const EncInfo, *mut *mut c_void) -> i32;
    type FnEncode = unsafe extern "C" fn(*mut c_void, *const CpuBuf, *const Rate) -> i32;
    type FnNum = unsafe extern "C" fn(*mut c_void, usize, *mut usize) -> i32;
    type FnPacket = unsafe extern "C" fn(*mut c_void, *mut Pkt, usize, *mut usize, *mut u8, usize) -> i32;
    type FnDestroy = unsafe extern "C" fn(*mut c_void);

    struct Api {
        mk: FnDefault,
        compat: FnCompat,
        enc_create: FnEncCreate,
        encode: FnEncode,
        num: FnNum,
        packet: FnPacket,
        enc_destroy: FnDestroy,
        dev_destroy: FnDestroy,
    }

    fn sym<T>(module: *mut c_void, name: &str) -> Result<T, String> {
        let mut z = name.as_bytes().to_vec();
        z.push(0);
        let ptr = unsafe { GetProcAddress(module, z.as_ptr().cast()) };
        if ptr.is_null() {
            return Err(format!("GetProcAddress {name} failed {}", unsafe { GetLastError() }));
        }
        Ok(unsafe { std::mem::transmute_copy(&ptr) })
    }

    fn load() -> Result<Api, String> {
        let dir = std::env::var("PYROWAVE_DIR").unwrap_or_else(|_| r"C:\Program Files\etherquic".into());
        let dll = std::env::var("PYROWAVE_DLL").unwrap_or_else(|_| format!(r"{dir}\libpyrowave-shared-1.dll"));
        let mut dir_z = dir.into_bytes();
        dir_z.push(0);
        unsafe { SetDllDirectoryA(dir_z.as_ptr()) };
        let mut vk = b"vulkan-1.dll\0".to_vec();
        let vk_mod = unsafe { LoadLibraryA(vk.as_mut_ptr()) };
        if vk_mod.is_null() {
            return Err(format!("LoadLibrary vulkan-1.dll failed {}", unsafe { GetLastError() }));
        }
        let mut path = dll.into_bytes();
        path.push(0);
        let module = unsafe { LoadLibraryA(path.as_ptr()) };
        if module.is_null() {
            return Err(format!(
                "LoadLibrary libpyrowave-shared-1.dll failed {}",
                unsafe { GetLastError() }
            ));
        }
        Ok(Api {
            mk: sym(module, "pyrowave_create_default_device")?,
            compat: sym(module, "pyrowave_create_device_by_compat")?,
            enc_create: sym(module, "pyrowave_encoder_create")?,
            encode: sym(module, "pyrowave_encoder_encode_cpu")?,
            num: sym(module, "pyrowave_encoder_compute_num_packets")?,
            packet: sym(module, "pyrowave_encoder_packetize")?,
            enc_destroy: sym(module, "pyrowave_encoder_destroy")?,
            dev_destroy: sym(module, "pyrowave_device_destroy")?,
        })
    }

    pub fn encode(y: &[u8], u: &[u8], v: &[u8], w: u32, h: u32, target: u32) -> Result<Vec<u8>, String> {
        let api = load()?;
        unsafe {
            let mut dev = std::ptr::null_mut();
            // VK_QUEUE_GLOBAL_PRIORITY_MEDIUM = 256. Prefer the NVIDIA device so the
            // oracle is the RTX, not the UHD 770.
            let rc = (api.compat)(0x10de, 0, std::ptr::null(), std::ptr::null(), std::ptr::null(), 256, &mut dev);
            if rc != 0 || dev.is_null() {
                let rc_default = (api.mk)(&mut dev);
                if rc_default != 0 || dev.is_null() {
                    return Err(format!(
                        "pyrowave_create_device_by_compat(10de) failed {rc}; pyrowave_create_default_device failed {rc_default}"
                    ));
                }
                println!("ORIGINAL_DEVICE via=default compat_rc={rc}");
            } else {
                println!("ORIGINAL_DEVICE via=compat vid=10de");
            }
            let info = EncInfo { device: dev, width: w as i32, height: h as i32, chroma: 0 };
            let mut enc = std::ptr::null_mut();
            let rc = (api.enc_create)(&info, &mut enc);
            if rc != 0 || enc.is_null() {
                (api.dev_destroy)(dev);
                return Err(format!("pyrowave_encoder_create failed {rc}"));
            }
            let buf = CpuBuf {
                data: [y.as_ptr() as *mut c_void, u.as_ptr() as *mut c_void, v.as_ptr() as *mut c_void],
                row_stride: [w as usize, (w / 2) as usize, (w / 2) as usize],
                plane_size: [y.len(), u.len(), v.len()],
                width: w as i32,
                height: h as i32,
                format: 1,
            };
            let rate = Rate { maximum_bitstream_size: target as usize };
            let rc = (api.encode)(enc, &buf, &rate);
            if rc != 0 {
                (api.enc_destroy)(enc);
                (api.dev_destroy)(dev);
                return Err(format!("pyrowave_encoder_encode_cpu failed {rc}"));
            }
            let boundary = (16usize << 20).max(target as usize);
            let mut n = 0usize;
            let rc = (api.num)(enc, boundary, &mut n);
            if rc != 0 {
                (api.enc_destroy)(enc);
                (api.dev_destroy)(dev);
                return Err(format!("pyrowave_encoder_compute_num_packets failed {rc}"));
            }
            let mut pk = vec![Pkt { offset: 0, size: 0 }; n.max(1)];
            let mut bits = vec![0u8; boundary];
            let mut np = 0usize;
            let rc = (api.packet)(enc, pk.as_mut_ptr(), boundary, &mut np, bits.as_mut_ptr(), bits.len());
            (api.enc_destroy)(enc);
            (api.dev_destroy)(dev);
            if rc != 0 {
                return Err(format!("pyrowave_encoder_packetize failed {rc}"));
            }
            if np == 0 || np > pk.len() {
                return Err(format!("packetize returned {np} packets"));
            }
            let mut walked = 0usize;
            for packet in pk.iter().take(np) {
                if packet.offset != walked {
                    return Err(format!("original packet gap at {walked}"));
                }
                walked += packet.size;
            }
            bits.truncate(walked);
            Ok(bits)
        }
    }
}

#[cfg(not(windows))]
mod original {
    pub fn encode(_y: &[u8], _u: &[u8], _v: &[u8], _w: u32, _h: u32, _target: u32) -> Result<Vec<u8>, String> {
        Err("original Vulkan encoder is the Windows libpyrowave-shared-1.dll".into())
    }
}

fn one(
    backend: &str,
    which: Option<Backend>,
    y: &[u8],
    u: &[u8],
    v: &[u8],
    w: u32,
    h: u32,
    target: u32,
    oracle: Option<&[u8]>,
) -> Option<Vec<u8>> {
    let encoded = if let Some(kind) = which {
        encode_rust(kind, y, u, v, w, h, target)
    } else {
        original::encode(y, u, v, w, h, target)
    };
    match encoded {
        Ok(bits) => {
            let cmp = if backend == "original" {
                "cmp=oracle".to_string()
            } else if let Some(oracle) = oracle {
                compare(backend, &bits, oracle)
            } else {
                "cmp=no-original".to_string()
            };
            record(backend, w, h, target, &bits, &cmp);
            Some(bits)
        }
        Err(err) => {
            println!("RECORD backend={backend} W={w} H={h} target={target} ERROR={err}");
            None
        }
    }
}

fn frame(w: u32, h: u32, targets: &[u32]) {
    let y = plane(w as usize, h as usize, 1);
    let u = plane(w as usize / 2, h as usize / 2, 2);
    let v = plane(w as usize / 2, h as usize / 2, 3);
    let backends = [
        ("original", None),
        ("cpu", Some(Backend::Cpu)),
        ("vulkan", Some(Backend::Vulkan)),
        ("d3d11", Some(Backend::D3d11)),
        ("opengl", Some(Backend::OpenGl)),
    ];
    let mut by_backend: Vec<(&str, Vec<Option<usize>>)> = Vec::new();
    for (name, _) in backends {
        by_backend.push((name, Vec::new()));
    }
    for &target in targets {
        let mut oracle = None;
        for (i, (name, kind)) in backends.iter().copied().enumerate() {
            let bits = one(name, kind, &y, &u, &v, w, h, target, oracle.as_deref());
            if name == "original" {
                oracle = bits.clone();
            }
            by_backend[i].1.push(bits.as_ref().map(|b| b.len()));
        }
    }
    for (name, lens) in &by_backend {
        if lens.len() == 2 {
            match (lens[0], lens[1]) {
                (Some(hi), Some(lo)) => {
                    let moved = u8::from(lo < hi);
                    println!("TARGET_MOVE backend={name} W={w} H={h} hi_bytes={hi} lo_bytes={lo} moved={moved}");
                }
                _ => println!("TARGET_MOVE backend={name} W={w} H={h} missing=1"),
            }
        }
    }
}

fn main() {
    let mut args = env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "all".into());
    println!("same-rdo mode={mode}");
    if mode == "all" {
        for (w, h) in [(1280u32, 720u32), (1920, 1080)] {
            let hi = Encoder::default_target(w, h);
            let lo = (hi / 4) & !3;
            println!("TARGETS W={w} H={h} hi={hi} lo={lo}");
            frame(w, h, &[hi, lo]);
        }
        return;
    }
    let w: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    let h: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    let t0: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1 << 20);
    let t1: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(4096);
    println!("TARGETS W={w} H={h} hi={t0} lo={t1}");
    let only = match mode.as_str() {
        "cpu" => Some(Backend::Cpu),
        "vulkan" | "vk" => Some(Backend::Vulkan),
        "d3d11" | "dx" => Some(Backend::D3d11),
        "opengl" | "gl" => Some(Backend::OpenGl),
        "original" => None,
        _ => Some(Backend::Cpu),
    };
    let y = plane(w as usize, h as usize, 1);
    let u = plane(w as usize / 2, h as usize / 2, 2);
    let v = plane(w as usize / 2, h as usize / 2, 3);
    for target in [t0, t1] {
        let oracle = if mode != "original" { original::encode(&y, &u, &v, w, h, target).ok() } else { None };
        if mode == "original" {
            one("original", None, &y, &u, &v, w, h, target, None);
        } else {
            one(&mode, only, &y, &u, &v, w, h, target, oracle.as_deref());
        }
    }
}
