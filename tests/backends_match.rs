//! One 4:2:0 frame through `Encoder::create` on every backend.
//! CPU must emit a PWV1 file. Vulkan (software devices allowed, so lavapipe
//! is eligible) must match those packet bytes or return the loader's own
//! error. D3D11 and OpenGL call the real constructors the same way.

use pyrowave_rs::{pwv1_one_frame, Backend, Chroma, CreateInfo, Encoder, YuvFrame};
use std::ffi::CString;
use std::os::raw::c_char;

fn pin_lavapipe_icd() {
    extern "C" {
        fn getenv(name: *const c_char) -> *const c_char;
        fn setenv(name: *const c_char, value: *const c_char, overwrite: i32) -> i32;
    }
    let Ok(files) = CString::new("VK_DRIVER_FILES") else {
        return;
    };
    let Ok(icd) = CString::new("VK_ICD_FILENAMES") else {
        return;
    };
    unsafe {
        if !getenv(files.as_ptr()).is_null() || !getenv(icd.as_ptr()).is_null() {
            return;
        }
    }
    let dir = std::path::Path::new("/run/opengl-driver/share/vulkan/icd.d");
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let Some(path) = rd.flatten().map(|ent| ent.path()).find(|p| {
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("").to_ascii_lowercase();
        name.contains("lvp") && name.ends_with(".json")
    }) else {
        return;
    };
    let Ok(val) = CString::new(path.to_string_lossy().as_ref()) else {
        return;
    };
    unsafe {
        let _ = setenv(files.as_ptr(), val.as_ptr(), 0);
    }
}

fn planes() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; 64 * 64];
    let mut u = vec![0u8; 32 * 32];
    let mut v = vec![0u8; 32 * 32];
    for i in 0..y.len() {
        y[i] = ((i * 3 + (i / 64) * 5) & 255) as u8;
    }
    for i in 0..u.len() {
        u[i] = (96 + (i & 31)) as u8;
        v[i] = (160 + (i * 2 & 31)) as u8;
    }
    (y, u, v)
}

fn payload(backend: Backend, allow_software: bool) -> Result<(String, Vec<u8>), String> {
    let mut enc = Encoder::create(CreateInfo { backend, allow_software })?;
    let name = enc.device_name.clone();
    let (y, u, v) = planes();
    enc.encode(&YuvFrame {
        width: 64,
        height: 64,
        chroma: Chroma::Yuv420,
        y: &y,
        u: &u,
        v: &v,
    })?;
    let boundary = 1 << 20;
    let mut bits = vec![0u8; boundary];
    let packets = enc.packetize(boundary, &mut bits)?;
    assert!(!packets.is_empty(), "{backend:?} produced no packets");
    let len: usize = packets.iter().map(|p| p.size).sum();
    assert!(len > 0, "{backend:?} packet payload is empty");
    bits.truncate(len);
    Ok((name, bits))
}

#[test]
fn cpu_420_is_a_pwv1_sequence() {
    let (_name, bits) = payload(Backend::Cpu, false).expect("cpu encode");
    let file = pwv1_one_frame(64, 64, 24, 1, &bits);
    assert!(file.starts_with(b"PWV1"), "packetizer header is PWV1");
    assert!(file.len() > 28, "header plus a non-empty frame");
    let declared = u32::from_le_bytes(file[24..28].try_into().unwrap()) as usize;
    assert_eq!(declared, bits.len());
    assert_eq!(&file[28..], bits.as_slice());
    eprintln!("CPU-PWV1 bytes={} payload={}", file.len(), bits.len());
}

#[test]
fn vulkan_lavapipe_matches_cpu_or_reports_the_loader() {
    // NixOS does not put the loader or ICDs on the default path. Selecting the
    // lavapipe JSON here keeps the NVIDIA ICD from winning the device pick.
    pin_lavapipe_icd();
    let cpu = payload(Backend::Cpu, false).expect("cpu encode").1;
    match payload(Backend::Vulkan, true) {
        Ok((name, bits)) => {
            assert!(!name.is_empty(), "Vulkan device has no name");
            let lower = name.to_ascii_lowercase();
            assert!(
                lower.contains("lavapipe") || lower.contains("llvmpipe"),
                "software Vulkan device must be lavapipe, got {name}"
            );
            assert_eq!(bits, cpu, "lavapipe packets differ from CPU");
            eprintln!("LAVAPIPE-MATCH device={name} bytes={}", bits.len());
        }
        Err(err) => {
            assert!(
                err.contains("Vulkan") || err.contains("libvulkan") || err.contains("cannot open"),
                "Vulkan failure must be the loader's own text: {err}"
            );
            eprintln!("LAVAPIPE-ERROR {err}");
        }
    }
}

#[test]
fn d3d11_and_opengl_constructors_match_cpu_or_report_the_loader() {
    let cpu = payload(Backend::Cpu, false).expect("cpu encode").1;
    for backend in [Backend::D3d11, Backend::OpenGl] {
        match payload(backend, false) {
            Ok((name, bits)) => {
                assert!(!name.is_empty(), "{backend:?} device has no name");
                assert_eq!(bits, cpu, "{backend:?} packets differ from CPU");
                eprintln!("GPU-CREATE {backend:?} ok device={name} bytes={}", bits.len());
            }
            Err(err) => {
                assert!(
                    err.contains("cannot open shared object")
                        || err.contains("LoadLibrary")
                        || err.contains("GetLastError")
                        || err.contains("software renderer")
                        || err.contains("context failed")
                        || err.contains("missing ")
                        || err.contains("device create failed"),
                    "{backend:?} failure must be the constructor's own text: {err}"
                );
                eprintln!("GPU-CREATE {backend:?} error: {err}");
            }
        }
    }
}
