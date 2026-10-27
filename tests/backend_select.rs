use pyrowave_rs::{
    d3d11_feature_level, d3d11_shader_target, d3d11_shaders_are_cs5, opengl_shaders_are_430,
    probe_gpu_devices, software_rejected, warp_rejected, Backend, Chroma, CreateInfo, Encoder,
    FEATURE_LEVEL_11_0, GL_MAJOR, GL_MINOR, SHADER_TARGET, YuvFrame,
};

#[test]
fn cpu_needs_no_gpu_and_gpu_predicates_match_the_ports() {
    let mut enc = Encoder::create(CreateInfo {
        backend: Backend::Cpu,
        allow_software: false,
    })
    .expect("cpu");
    assert!(!enc.uses_gpu());
    assert!(enc.device_name.is_empty());
    let y = vec![16u8; 64 * 64];
    let c = vec![128u8; 32 * 32];
    enc.encode(&YuvFrame {
        width: 64,
        height: 64,
        chroma: Chroma::Yuv420,
        y: &y,
        u: &c,
        v: &c,
    })
    .expect("cpu encode");

    assert_eq!(d3d11_feature_level(), FEATURE_LEVEL_11_0);
    assert_eq!(FEATURE_LEVEL_11_0, 0xb000);
    assert_eq!(d3d11_shader_target(), "cs_5_0");
    assert_eq!(SHADER_TARGET, "cs_5_0");
    assert!(d3d11_shaders_are_cs5());
    assert_eq!((GL_MAJOR, GL_MINOR), (4, 3));
    assert!(opengl_shaders_are_430());

    assert!(warp_rejected(0x1414, 0x008c, "Microsoft Basic Render Driver", false));
    assert!(!warp_rejected(0x1414, 0x008c, "Microsoft Basic Render Driver", true));
    assert!(warp_rejected(0x10de, 0x2504, "WARP", false));
    assert!(!warp_rejected(0x10de, 0x2504, "NVIDIA GeForce RTX 3060", false));

    assert!(software_rejected("llvmpipe (LLVM 23.1.2, 512 bits)", false));
    assert!(software_rejected("softpipe", false));
    assert!(software_rejected("SwiftShader", false));
    assert!(!software_rejected("llvmpipe (LLVM 23.1.2, 512 bits)", true));
    assert!(!software_rejected("NVIDIA GeForce RTX 3060/PCIe/SSE2", false));

    let d3d = Encoder::create(CreateInfo {
        backend: Backend::D3d11,
        allow_software: false,
    });
    match &d3d {
        Err(err) => assert!(
            err.contains("cannot open shared object") || err.contains("LoadLibrary") || err.contains("GetLastError"),
            "D3D11 failure must be the loader's own text: {err}"
        ),
        Ok(enc) => assert!(!enc.device_name.is_empty(), "a real adapter has a name"),
    }
    let gl = Encoder::create(CreateInfo {
        backend: Backend::OpenGl,
        allow_software: false,
    });
    // A hardware context may succeed. A missing loader or a software renderer
    // must come back as an error from the real constructor.
    if let Err(err) = &gl {
        assert!(
            err.contains("cannot open shared object")
                || err.contains("software renderer")
                || err.contains("context failed")
                || err.contains("missing "),
            "OpenGL failure must be the loader or the context: {err}"
        );
    }
}

#[test]
fn gpu_probe_writes_real_constructor_text() {
    let (d3d, gl) = probe_gpu_devices(false);
    assert!(d3d.contains("11_0") || d3d.contains("0xb000") || d3d.contains("cs_5_0"), "{d3d}");
    assert!(gl.contains("OpenGL") || gl.starts_with("OpenGL ok"), "{gl}");
    eprintln!("D3D11: {d3d}");
    eprintln!("GL: {gl}");
}
