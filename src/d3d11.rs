//! Direct3D 11 backend. Device creation loads `d3d11.dll` and calls
//! `D3D11CreateDevice` with feature level 11_0 only. The compute programs are
//! the subgroup-free `cs_5_0` shaders from `src/shaders`. WARP (vendor 0x1414
//! device 0x008C, or a name containing "warp" / "basic") is released unless
//! `allow_software` is set.

use crate::d3d11_plan::{self, Submission, FORMAT_R32_FLOAT, FORMAT_R32_SINT, FORMAT_R32_TYPELESS, FORMAT_R32_UINT, FORMAT_R8_UNORM, SRV_DIM_BUFFEREX, SRV_DIM_TEXTURE2DARRAY, UAV_DIM_BUFFER, UAV_DIM_TEXTURE2DARRAY};
use crate::packet::{BlockMeta, FrameBits};
use crate::rdo::LevelWave;
use crate::Chroma;

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};

/// `D3D_FEATURE_LEVEL_11_0`.
pub const FEATURE_LEVEL_11_0: u32 = 0xb000;
pub const SHADER_TARGET: &str = "cs_5_0";

pub const DWT_HLSL: &str = include_str!("../shaders/ext/dwt.hlsl");
pub const QUANT_HLSL: &str = include_str!("../shaders/ext/quant.hlsl");
pub const PACK_HLSL: &str = include_str!("../shaders/ext/pack.hlsl");
/// fxc `/T cs_5_0 /E main /O3 /Ges /Gis` output. Same flags as the C++ port.
const DWT_DXBC: &[u8] = include_bytes!("dwt.dxbc");
const QUANT_DXBC: &[u8] = include_bytes!("quant.dxbc");
const PACK_DXBC: &[u8] = include_bytes!("pack.dxbc");
const COMMON_HLSLI: &str = include_str!("../shaders/ext/common.hlsli");
const PROBE_FS_Q2: &str = include_str!("../shaders/ext/probe_fs_q2.inc");

const D3D11_SDK_VERSION: u32 = 7;
const D3D_DRIVER_TYPE_UNKNOWN: u32 = 0;
const D3D_DRIVER_TYPE_HARDWARE: u32 = 1;
const D3DCOMPILE_ENABLE_STRICTNESS: u32 = 1 << 11;
const D3DCOMPILE_IEEE_STRICTNESS: u32 = 1 << 13;
const D3DCOMPILE_OPTIMIZATION_LEVEL3: u32 = 1 << 15;
const D3DCOMPILE_WARNINGS_ARE_ERRORS: u32 = 1 << 18;

pub fn embedded_shader_targets() -> &'static [&'static str] {
    const TARGETS: [&str; 3] = [SHADER_TARGET, SHADER_TARGET, SHADER_TARGET];
    &TARGETS
}

pub fn is_warp_adapter(vendor_id: u32, device_id: u32, name: &str) -> bool {
    if vendor_id == 0x1414 && device_id == 0x008c {
        return true;
    }
    let lower = name.to_ascii_lowercase();
    lower.contains("warp") || lower.contains("basic")
}

/// True when this adapter must be skipped. WARP is accepted only with the flag.
pub fn warp_rejected(vendor_id: u32, device_id: u32, name: &str, allow_software: bool) -> bool {
    is_warp_adapter(vendor_id, device_id, name) && !allow_software
}

pub fn shader_sources_are_cs5() -> bool {
    embedded_shader_targets().iter().all(|target| *target == SHADER_TARGET)
        && DWT_HLSL.contains("cs_5_0")
        && QUANT_HLSL.contains("cs_5_0")
        && PACK_HLSL.contains("cs_5_0")
        && !DWT_HLSL.contains("cs_6_0")
}

/// The embedded programs encode compiles. Texture2DArray planes at t3 and a
/// 192-byte `jobw[12]` cbuffer, matching `dwt.hlsl`.
pub fn shader_layout_matches_ports() -> bool {
    DWT_HLSL.contains("Texture2DArray<float> uPlane0 : register(t3)")
        && DWT_HLSL.contains("RWTexture2DArray<float> uOutput : register(u0)")
        && DWT_HLSL.contains("RWBuffer<float> uCoeff : register(u1)")
        && DWT_HLSL.contains("uint4 jobw[12]")
        && QUANT_HLSL.contains("uint4 w[10]")
        && QUANT_HLSL.contains("ByteAddressBuffer t0 : register(t0)")
        && PACK_HLSL.contains("ByteAddressBuffer tJobs : register(t3)")
}

#[repr(C)]
struct Guid {
    data1: u32,
    data2: u16,
    data3: u16,
    data4: [u8; 8],
}

// IDXGIDevice
const IID_IDXGI_DEVICE: Guid = Guid {
    data1: 0x54ec77fa,
    data2: 0x1377,
    data3: 0x44e6,
    data4: [0x8c, 0x32, 0x88, 0xfd, 0x5f, 0x44, 0xc8, 0x4c],
};

struct Live {
    name: String,
    d3d11: *mut c_void,
    compiler: *mut c_void,
    device: *mut c_void,
    context: *mut c_void,
    shaders: [*mut c_void; 3],
    wavelet: *mut c_void,
}

struct WaveGpu {
    src: *mut c_void,
    src_uav: *mut c_void,
    src_up: *mut c_void,
    dst: *mut c_void,
    dst_uav: *mut c_void,
    dst_up: *mut c_void,
    stage: *mut c_void,
    cb: *mut c_void,
    scratch: *mut c_void,
    scratch_uav: *mut c_void,
    cap_in: usize,
    cap_out: usize,
    cap_scratch: usize,
}

pub struct Device {
    live: Box<Live>,
    pyramid: std::cell::RefCell<Option<Pyramid>>,
    wave: std::cell::RefCell<Option<WaveGpu>>,
    /// Sample count of a dispatched level whose readback has not run yet. Zero means idle.
    wave_inflight: std::cell::Cell<usize>,
}

unsafe impl Send for Device {}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            self.pyramid.borrow_mut().take();
            if let Some(wave) = self.wave.borrow_mut().take() {
                release(wave.src_uav);
                release(wave.dst_uav);
                release(wave.src_up);
                release(wave.dst_up);
                release(wave.src);
                release(wave.dst);
                release(wave.scratch_uav);
                release(wave.scratch);
                release(wave.stage);
                release(wave.cb);
            }
            release(self.live.wavelet);
            release(self.live.context);
            for shader in self.live.shaders {
                release(shader);
            }
            release(self.live.device);
            free_library(self.live.compiler);
            free_library(self.live.d3d11);
        }
    }
}

pub fn device_name(device: &Device) -> &str {
    &device.live.name
}

/// Load `d3d11.dll`, create a feature-level 11_0 device, and compile the
/// embedded DWT, quant, and pack shaders as `cs_5_0`.
pub fn create_device(allow_software: bool) -> Result<Device, String> {
    let levels = [FEATURE_LEVEL_11_0];
    unsafe { create_device_inner(allow_software, &levels) }
}

unsafe fn create_device_inner(allow_software: bool, levels: &[u32]) -> Result<Device, String> {
    let d3d11 = open_library("d3d11.dll").map_err(|detail| {
        format!(
            "D3D11 device create failed: {detail}; requested feature level {:#x} ({SHADER_TARGET})",
            levels[0]
        )
    })?;
    let create: CreateDevice = match lib_sym(d3d11, "D3D11CreateDevice") {
        Ok(f) => f,
        Err(err) => {
            free_library(d3d11);
            return Err(format!("D3D11 device create failed: {err}"));
        }
    };

    let mut device = std::ptr::null_mut();
    let mut context = std::ptr::null_mut();
    let mut got = 0u32;
    let adapter = preferred_adapter();
    let driver = if adapter.is_null() { D3D_DRIVER_TYPE_HARDWARE } else { D3D_DRIVER_TYPE_UNKNOWN };
    let hr = create(
        adapter,
        driver,
        std::ptr::null_mut(),
        // D3D11_CREATE_DEVICE_SINGLETHREADED. The encoder calls the device on one thread.
        1,
        levels.as_ptr(),
        levels.len() as u32,
        D3D11_SDK_VERSION,
        &mut device,
        &mut got,
        &mut context,
    );
    release(adapter);
    if hr < 0 || device.is_null() || context.is_null() || got < FEATURE_LEVEL_11_0 {
        release(context);
        release(device);
        free_library(d3d11);
        return Err(format!(
            "D3D11CreateDevice failed: hr={hr:#x} feature level {got:#x}, requested {:#x}",
            levels[0]
        ));
    }
    let _ = D3D_DRIVER_TYPE_UNKNOWN;

    let (vendor, device_id, name) = match adapter_identity(device) {
        Ok(v) => v,
        Err(err) => {
            release(context);
            release(device);
            free_library(d3d11);
            return Err(err);
        }
    };
    if warp_rejected(vendor, device_id, &name, allow_software) {
        release(context);
        release(device);
        free_library(d3d11);
        return Err(format!("D3D11 device create failed: WARP adapter rejected: {name}"));
    }

    let compiler = match open_library("d3dcompiler_47.dll") {
        Ok(h) => h,
        Err(err) => {
            release(context);
            release(device);
            free_library(d3d11);
            return Err(format!("D3D11 device create failed: {err}"));
        }
    };
    let compile: CompileFn = match lib_sym(compiler, "D3DCompile") {
        Ok(f) => f,
        Err(err) => {
            free_library(compiler);
            release(context);
            release(device);
            free_library(d3d11);
            return Err(format!("D3D11 device create failed: {err}"));
        }
    };
    let blobs = [("dwt.dxbc", DWT_DXBC), ("quant.dxbc", QUANT_DXBC), ("pack.dxbc", PACK_DXBC)];
    let mut shaders = [std::ptr::null_mut(); 3];
    for (i, (file, bytes)) in blobs.iter().enumerate() {
        match create_cs_bytes(device, file, bytes) {
            Ok(shader) => shaders[i] = shader,
            Err(err) => {
                for shader in shaders {
                    release(shader);
                }
                free_library(compiler);
                release(context);
                release(device);
                free_library(d3d11);
                return Err(err);
            }
        }
    }
    // TeraScale 2 miscompiles the integer f16 helpers. Hardware opcodes match
    // on values that already live in a register (probed on the HD 6670).
    let wavelet_src = if vendor == 0x1002 {
        format!("#define PYRO_AMD 1\n{RDO_DWT_HLSL}")
    } else {
        RDO_DWT_HLSL.to_string()
    };
    let wavelet = match compile_cs5(device, compile, "rdo_dwt.hlsl", &wavelet_src) {
        Ok(shader) => shader,
        Err(err) => {
            for shader in shaders {
                release(shader);
            }
            free_library(compiler);
            release(context);
            release(device);
            free_library(d3d11);
            return Err(format!("D3D11 device create failed: {err}"));
        }
    };

    Ok(Device {
        live: Box::new(Live {
            name,
            d3d11,
            compiler,
            device,
            context,
            shaders,
            wavelet,
        }),
        pyramid: std::cell::RefCell::new(None),
        wave: std::cell::RefCell::new(None),
        wave_inflight: std::cell::Cell::new(0),
    })
}

const RDO_DWT_HLSL: &str = include_str!("../shaders/rdo_dwt.hlsl");

unsafe fn preferred_adapter() -> *mut c_void {
    let dxgi = match open_library("dxgi.dll") {
        Ok(handle) => handle,
        Err(_) => return std::ptr::null_mut(),
    };
    type CreateFactory = unsafe extern "system" fn(*const Guid, *mut *mut c_void) -> i32;
    let create: CreateFactory = match lib_sym(dxgi, "CreateDXGIFactory1") {
        Ok(f) => f,
        Err(_) => return std::ptr::null_mut(),
    };
    let iid = Guid {
        data1: 0x770a_ae78,
        data2: 0xf26f,
        data3: 0x4dba,
        data4: [0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87],
    };
    let mut factory = std::ptr::null_mut();
    if create(&iid, &mut factory) < 0 || factory.is_null() {
        return std::ptr::null_mut();
    }
    type EnumAdapters = unsafe extern "system" fn(*mut c_void, u32, *mut *mut c_void) -> i32;
    type GetDesc = unsafe extern "system" fn(*mut c_void, *mut AdDesc) -> i32;
    let enum_adapters: EnumAdapters = vslot(factory, 7);
    let mut chosen = std::ptr::null_mut();
    let mut index = 0u32;
    loop {
        let mut adapter = std::ptr::null_mut();
        if enum_adapters(factory, index, &mut adapter) < 0 || adapter.is_null() {
            break;
        }
        let get_desc: GetDesc = vslot(adapter, 8);
        let mut desc = std::mem::zeroed::<AdDesc>();
        let hr = get_desc(adapter, &mut desc);
        if hr >= 0 && desc.vendor == 0x10de {
            chosen = adapter;
            break;
        }
        release(adapter);
        index += 1;
        if index > 16 {
            break;
        }
    }
    release(factory);
    let _ = dxgi;
    chosen
}

#[repr(C)]
struct AdDesc {
    description: [u16; 128],
    vendor: u32,
    device: u32,
    subsys: u32,
    revision: u32,
    dedicated_video: usize,
    dedicated_system: usize,
    shared_system: usize,
    luid_low: u32,
    luid_high: i32,
}

/// Dispatch this device's cs_5_0 wavelet, then the shared quant/analyze/resolve/RDO path.
pub fn encode(
    device: &Device,
    scratch: &mut crate::rdo::Scratch,
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: u32,
    height: u32,
    chroma: Chroma,
    target: u32,
    sequence: u32,
) -> Result<FrameBits, String> {
    let mut wave = DxWave { device };
    crate::rdo::encode_scratch(scratch, y, u, v, width, height, chroma, target, sequence, |src, rw, rh, alw, alh, dc, out| {
        wave.dispatch(src, rw, rh, alw, alh, dc, out)
    })
    .map_err(|err| format!("D3D11 encode failed: {err}"))
}

pub fn encode_pair(
    device: &Device,
    slot0: &mut crate::rdo::Scratch,
    slot1: &mut crate::rdo::Scratch,
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
) -> Result<(FrameBits, FrameBits), String> {
    let mut wave = DxWave { device };
    crate::rdo::encode_pair_scratch(
        slot0, slot1, y0, u0, v0, y1, u1, v1, width, height, chroma, target, seq0, seq1, &mut wave,
    )
    .map_err(|err| format!("D3D11 encode failed: {err}"))
}

struct DxWave<'a> {
    device: &'a Device,
}

impl crate::rdo::LevelWave for DxWave<'_> {
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
        unsafe {
            dwt_submit(self.device, src, res_w, res_h, aligned_w, aligned_h, dc)?;
            dwt_finish(self.device, out)
        }
    }

    fn submit_first(
        &mut self,
        src: &[f32],
        res_w: i32,
        res_h: i32,
        aligned_w: i32,
        aligned_h: i32,
        dc: bool,
    ) -> Result<(), String> {
        unsafe { dwt_submit(self.device, src, res_w, res_h, aligned_w, aligned_h, dc) }
    }

    fn finish_first(
        &mut self,
        _src: &[f32],
        _res_w: i32,
        _res_h: i32,
        _aligned_w: i32,
        _aligned_h: i32,
        _dc: bool,
        out: &mut [Vec<f32>; 4],
    ) -> Result<(), String> {
        unsafe { dwt_finish(self.device, out) }
    }
}

unsafe fn dwt_submit(
    device: &Device,
    src: &[f32],
    res_w: i32,
    res_h: i32,
    aligned_w: i32,
    aligned_h: i32,
    dc: bool,
) -> Result<(), String> {
    if device.wave_inflight.get() != 0 {
        return Err("D3D11 wavelet is already in flight".into());
    }
    let ow = aligned_w / 2;
    let oh = aligned_h / 2;
    let groups_x = (aligned_w + 31) / 32;
    let groups_y = (aligned_h + 31) / 32;
    if ow <= 0 || oh <= 0 || res_w <= 0 || res_h <= 0 {
        return Err("wavelet level is empty".into());
    }
    let need = (res_w as usize).saturating_mul(res_h as usize);
    if src.len() < need {
        return Err("wavelet source is short".into());
    }
    let n = (ow as usize).saturating_mul(oh as usize);
    let in_bytes = need * 4;
    let out_bytes = n * 4 * 4;
    let scratch_bytes = (groups_x as usize) * (groups_y as usize) * 884 * 4;
    ensure_wave(device, in_bytes, out_bytes, scratch_bytes)?;
    let wave = device.wave.borrow();
    let wave = wave.as_ref().ok_or("D3D11 wavelet buffers were not created")?;
    let bytes = std::slice::from_raw_parts(src.as_ptr() as *const u8, in_bytes);
    // The source UAV keeps the first level's byte width. A null box reads that
    // whole width, so the upload allocation is `cap_in` and the tail is zero.
    let mut src_upload = vec![0u8; wave.cap_in];
    src_upload[..in_bytes].copy_from_slice(bytes);
    // HD 6670 drops UpdateSubresource on a UAV buffer. Map a staging buffer
    // and CopyResource instead. The destination staging stays zero so a missed
    // store cannot leak an uninitialized coefficient into the rate control.
    upload_staging(device.live.context, wave.src_up, wave.src, &src_upload)?;
    copy_resource(device.live.context, wave.dst, wave.dst_up);
    #[repr(C)]
    struct Cb {
        res_w: i32,
        res_h: i32,
        aligned_w: i32,
        aligned_h: i32,
        dc: i32,
        dst_w: i32,
        pad0: i32,
        pad1: i32,
    }
    let cb = Cb {
        res_w,
        res_h,
        aligned_w,
        aligned_h,
        dc: i32::from(dc),
        dst_w: ow,
        pad0: std::env::var("PYRO_DWT_PROBE").ok().and_then(|s| s.parse().ok()).unwrap_or(0),
        pad1: groups_x,
    };
    let cb_bytes = std::slice::from_raw_parts(&cb as *const Cb as *const u8, std::mem::size_of::<Cb>());
    // A constant buffer rejects a destination box. The resource is exactly 32 bytes.
    update_whole(device.live.context, wave.cb, cb_bytes);
    cs_set_shader(device.live.context, device.live.wavelet);
    cs_set_cb(device.live.context, wave.cb);
    cs_set_uavs(device.live.context, 0, &[wave.src_uav, wave.dst_uav, wave.scratch_uav]);
    dispatch(device.live.context, groups_x as u32, groups_y as u32, 1);
    cs_set_uavs(
        device.live.context,
        0,
        &[std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut()],
    );
    cs_set_cb(device.live.context, std::ptr::null_mut());
    cs_set_shader(device.live.context, std::ptr::null_mut());
    flush_context(device.live.context);
    device.wave_inflight.set(n);
    Ok(())
}

unsafe fn dwt_finish(device: &Device, out: &mut [Vec<f32>; 4]) -> Result<(), String> {
    let n = device.wave_inflight.get();
    if n == 0 {
        return Err("D3D11 wavelet is not in flight".into());
    }
    let wave = device.wave.borrow();
    let wave = wave.as_ref().ok_or("D3D11 wavelet buffers were not created")?;
    copy_resource(device.live.context, wave.stage, wave.dst);
    #[repr(C)]
    struct Mapped {
        data: *mut c_void,
        row: u32,
        depth: u32,
    }
    type Map = unsafe extern "system" fn(*mut c_void, *mut c_void, u32, u32, u32, *mut Mapped) -> i32;
    type Unmap = unsafe extern "system" fn(*mut c_void, *mut c_void, u32);
    let map: Map = vslot(device.live.context, 14);
    let unmap: Unmap = vslot(device.live.context, 15);
    let mut mapped = Mapped { data: std::ptr::null_mut(), row: 0, depth: 0 };
    let hr = map(device.live.context, wave.stage, 0, MAP_READ, 0, &mut mapped);
    if hr < 0 || mapped.data.is_null() {
        return Err(format!("Map(wavelet) failed: hr={hr:#x}"));
    }
    let raw = std::slice::from_raw_parts(mapped.data as *const f32, n * 4);
    for band in 0..4 {
        out[band].clear();
        out[band].extend_from_slice(&raw[band * n..(band + 1) * n]);
    }
    unmap(device.live.context, wave.stage, 0);
    device.wave_inflight.set(0);
    crate::rdo_dwt::note_level("d3d11", out);
    Ok(())
}

unsafe fn ensure_wave(device: &Device, in_bytes: usize, out_bytes: usize, scratch_bytes: usize) -> Result<(), String> {
    {
        let slot = device.wave.borrow();
        if let Some(wave) = slot.as_ref() {
            if wave.cap_in >= in_bytes && wave.cap_out >= out_bytes && wave.cap_scratch >= scratch_bytes {
                return Ok(());
            }
        }
    }
    if let Some(old) = device.wave.borrow_mut().take() {
        release(old.src_uav);
        release(old.dst_uav);
        release(old.src_up);
        release(old.dst_up);
        release(old.src);
        release(old.dst);
        release(old.scratch_uav);
        release(old.scratch);
        release(old.stage);
        release(old.cb);
    }
    let scratch_size = align16(scratch_bytes.max(256));
    let in_size = align16(in_bytes.max(256));
    let out_size = align16(out_bytes.max(256));
    let src = create_buffer(device.live.device, in_size, USAGE_DEFAULT, BIND_UNORDERED_ACCESS, 0, 0)?;
    let src_uav = match make_uav_buffer(device.live.device, src, FORMAT_R32_FLOAT, (in_size / 4) as u32, 0) {
        Ok(view) => view,
        Err(err) => {
            release(src);
            return Err(err);
        }
    };
    let dst = match create_buffer(device.live.device, out_size, USAGE_DEFAULT, BIND_UNORDERED_ACCESS, 0, 0) {
        Ok(buf) => buf,
        Err(err) => {
            release(src_uav);
            release(src);
            return Err(err);
        }
    };
    let dst_uav = match make_uav_buffer(device.live.device, dst, FORMAT_R32_FLOAT, (out_size / 4) as u32, 0) {
        Ok(view) => view,
        Err(err) => {
            release(dst);
            release(src_uav);
            release(src);
            return Err(err);
        }
    };
    let stage = match create_staging(device.live.device, out_size) {
        Ok(buf) => buf,
        Err(err) => {
            release(dst_uav);
            release(dst);
            release(src_uav);
            release(src);
            return Err(err);
        }
    };
    let src_up = match create_upload(device.live.device, in_size) {
        Ok(buf) => buf,
        Err(err) => {
            release(stage);
            release(dst_uav);
            release(dst);
            release(src_uav);
            release(src);
            return Err(err);
        }
    };
    let dst_up = match create_upload(device.live.device, out_size) {
        Ok(buf) => buf,
        Err(err) => {
            release(src_up);
            release(stage);
            release(dst_uav);
            release(dst);
            release(src_uav);
            release(src);
            return Err(err);
        }
    };
    let zeros = vec![0u8; out_size];
    if let Err(err) = upload_staging_only(device.live.context, dst_up, &zeros) {
        release(dst_up);
        release(src_up);
        release(stage);
        release(dst_uav);
        release(dst);
        release(src_uav);
        release(src);
        return Err(err);
    }
    let cb = match create_buffer(device.live.device, 32, USAGE_DEFAULT, BIND_CONSTANT_BUFFER, 0, 0) {
        Ok(buf) => buf,
        Err(err) => {
            release(dst_up);
            release(src_up);
            release(stage);
            release(dst_uav);
            release(dst);
            release(src_uav);
            release(src);
            return Err(err);
        }
    };
    let scratch = match create_buffer(device.live.device, scratch_size, USAGE_DEFAULT, BIND_UNORDERED_ACCESS, 0, 0) {
        Ok(buf) => buf,
        Err(err) => {
            release(cb);
            release(dst_up);
            release(src_up);
            release(stage);
            release(dst_uav);
            release(dst);
            release(src_uav);
            release(src);
            return Err(err);
        }
    };
    let scratch_uav = match make_uav_buffer(device.live.device, scratch, FORMAT_R32_UINT, (scratch_size / 4) as u32, 0) {
        Ok(view) => view,
        Err(err) => {
            release(scratch);
            release(cb);
            release(dst_up);
            release(src_up);
            release(stage);
            release(dst_uav);
            release(dst);
            release(src_uav);
            release(src);
            return Err(err);
        }
    };
    *device.wave.borrow_mut() = Some(WaveGpu {
        src,
        src_uav,
        src_up,
        dst,
        dst_uav,
        dst_up,
        stage,
        cb,
        scratch,
        scratch_uav,
        cap_in: in_size,
        cap_out: out_size,
        cap_scratch: scratch_size,
    });
    Ok(())
}

unsafe fn copy_resource(ctx: *mut c_void, dst: *mut c_void, src: *mut c_void) {
    type Copy = unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void);
    let copy: Copy = vslot(ctx, 47);
    copy(ctx, dst, src);
}

unsafe fn flush_context(ctx: *mut c_void) {
    type Flush = unsafe extern "system" fn(*mut c_void);
    let flush: Flush = vslot(ctx, 111);
    flush(ctx);
}

unsafe fn upload_staging(ctx: *mut c_void, staging: *mut c_void, dest: *mut c_void, bytes: &[u8]) -> Result<(), String> {
    upload_staging_only(ctx, staging, bytes)?;
    copy_resource(ctx, dest, staging);
    Ok(())
}

unsafe fn upload_staging_only(ctx: *mut c_void, staging: *mut c_void, bytes: &[u8]) -> Result<(), String> {
    #[repr(C)]
    struct Mapped {
        data: *mut c_void,
        row: u32,
        depth: u32,
    }
    type Map = unsafe extern "system" fn(*mut c_void, *mut c_void, u32, u32, u32, *mut Mapped) -> i32;
    type Unmap = unsafe extern "system" fn(*mut c_void, *mut c_void, u32);
    let map: Map = vslot(ctx, 14);
    let unmap: Unmap = vslot(ctx, 15);
    let mut mapped = Mapped { data: std::ptr::null_mut(), row: 0, depth: 0 };
    let hr = map(ctx, staging, 0, 2, 0, &mut mapped);
    if hr < 0 || mapped.data.is_null() {
        return Err(format!("Map(upload) failed: hr={hr:#x}"));
    }
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped.data as *mut u8, bytes.len());
    unmap(ctx, staging, 0);
    Ok(())
}

fn pyramid_encode(
    device: &Device,
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: u32,
    height: u32,
    chroma_444: bool,
    blocks: Vec<BlockMeta>,
    recycle: Vec<u8>,
) -> Result<FrameBits, String> {
    let mut slot = device.pyramid.borrow_mut();
    let reuse = slot.as_ref().is_some_and(|pyramid| {
        pyramid.width == width && pyramid.height == height && pyramid.chroma_444 == chroma_444
    });
    if !reuse {
        if let Some(old) = slot.take() {
            drop(old);
        }
        *slot = Some(unsafe { build_pyramid(device, y, u, v, width, height, chroma_444)? });
    }
    let pyramid = slot.as_mut().unwrap();
    unsafe { run_pyramid(device, pyramid, y, u, v, blocks, recycle) }
}

type MapFn = unsafe extern "system" fn(*mut c_void, *mut c_void, u32, u32, u32, *mut Mapped) -> i32;
type UnmapFn = unsafe extern "system" fn(*mut c_void, *mut c_void, u32);

struct DwtBind {
    srvs: [*mut c_void; 3],
    uavs: [*mut c_void; 6],
    cb: *mut c_void,
    gx: u32,
    gy: u32,
    gz: u32,
}

struct QuantBind {
    srv: *mut c_void,
    cb: *mut c_void,
    gx: u32,
    gy: u32,
    gz: u32,
}

struct Pyramid {
    width: u32,
    height: u32,
    chroma_444: bool,
    plan: d3d11_plan::Submission,
    guard: FrameGuard,
    dwt_cb: Vec<GpuBuf>,
    quant_cb: Vec<GpuBuf>,
    dwt_bind: Vec<DwtBind>,
    quant_bind: Vec<QuantBind>,
    stage: *mut c_void,
    copy_bits: u32,
    map_fn: MapFn,
    unmap_fn: UnmapFn,
}

impl Drop for Pyramid {
    fn drop(&mut self) {
        unsafe { release(self.stage); }
        for cb in self.dwt_cb.drain(..) {
            destroy_buf(cb);
        }
        for cb in self.quant_cb.drain(..) {
            destroy_buf(cb);
        }
    }
}

unsafe fn build_pyramid(
    device: &Device,
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: u32,
    height: u32,
    chroma_444: bool,
) -> Result<Pyramid, String> {
    let plan = d3d11_plan::submission_plan(width, height, chroma_444)?;
    let dev = device.live.device;
    let ctx = device.live.context;
    let mut guard = FrameGuard {
        ctx,
        planes: [None, None, None],
        bands: Default::default(),
        sampler: std::ptr::null_mut(),
        bufs: Vec::new(),
    };
    let planes_src = [y, u, v];
    for (i, (pw, ph)) in plan.plane_size.iter().enumerate() {
        let desc = d3d11_plan::plane_texture_desc(*pw, *ph);
        guard.planes[i] = Some(create_plane(dev, ctx, &desc, planes_src[i])?);
    }
    for level in 0..5u8 {
        for c in 0..3usize {
            let needed = plan.dwt.iter().any(|b| b.slots.iter().any(|s| s.component == c as u8 && s.level == level));
            if !needed {
                continue;
            }
            let bw = (plan.aligned_w >> (level as i32 + 1)).max(0) as u32;
            let bh = (plan.aligned_h >> (level as i32 + 1)).max(0) as u32;
            let desc = d3d11_plan::band_texture_desc(bw, bh);
            guard.bands[c][level as usize] = Some(create_band(dev, ctx, &desc)?);
        }
    }
    guard.sampler = create_sampler(dev)?;
    guard.bufs.push(create_uint_buffer(dev, (plan.total8.max(1) as usize) * 8, true, true)?);
    guard.bufs.push(create_uint_buffer(dev, (plan.total8.max(1) as usize) * 64, true, true)?);
    let payload_bytes = ((plan.aligned_w as usize) * (plan.aligned_h as usize) * 8
        + (plan.total8.max(0) as usize) * 64
        + 256
        + 15)
        & !15;
    guard.bufs.push(create_uint_buffer(dev, payload_bytes, true, true)?);
    guard.bufs.push(create_sint_buffer(dev, (plan.total32.max(1) as usize) * 4)?);
    guard.bufs.push(create_uint_buffer(dev, (plan.total32.max(1) as usize) * 640 * 4, true, false)?);
    guard.bufs.push(create_uint_buffer(dev, (plan.total32.max(1) as usize) * 8, true, false)?);
    guard.bufs.push(create_raw_srv_buffer(dev, &plan.pack_jobs)?);
    let mut dwt_cb = Vec::new();
    for batch in &plan.dwt {
        let cb = create_cb_bytes(dev, &batch.push)?;
        update_subresource(ctx, cb.buffer, &batch.push);
        dwt_cb.push(cb);
    }
    let mut quant_cb = Vec::new();
    for job in &plan.quant {
        let cb = create_cb_bytes(dev, &job.push)?;
        update_subresource(ctx, cb.buffer, &job.push);
        quant_cb.push(cb);
    }
    update_subresource(ctx, guard.bufs[6].buffer, &plan.pack_jobs);
    // Quant codes stay zero for the life of this size. frame.cpp clears that
    // UAV once; a stale nonzero code drops every plane.
    upload_zeros(ctx, &guard.bufs[3]);
    // Pack rewrites meta for every block it visits, including a zero store for
    // an empty block (`pack.hlsl`). One clear at creation covers the slots the
    // dispatch does not touch. Doing it again each frame stalled the 1080p run.
    upload_zeros(ctx, &guard.bufs[5]);
    // frame.cpp copies at most 128 KiB of bitstream plus the 16-byte payload
    // head and the meta. One staging buffer, one Map.
    let copy_bits = (guard.bufs[4].bytes as u32).min(131072);
    let stage = create_staging(dev, 16 + copy_bits as usize)?;
    let dwt_bind = bind_dwt(&guard, &plan, &dwt_cb)?;
    let quant_bind = bind_quant(&guard, &plan, &quant_cb)?;
    Ok(Pyramid {
        width,
        height,
        chroma_444,
        plan,
        guard,
        dwt_cb,
        quant_cb,
        dwt_bind,
        quant_bind,
        stage,
        copy_bits,
        map_fn: vslot(ctx, 14),
        unmap_fn: vslot(ctx, 15),
    })
}

unsafe fn bind_dwt(guard: &FrameGuard, plan: &d3d11_plan::Submission, cbs: &[GpuBuf]) -> Result<Vec<DwtBind>, String> {
    let mut out = Vec::with_capacity(plan.dwt.len());
    for (batch, cb) in plan.dwt.iter().zip(cbs.iter()) {
        let mut srvs = [std::ptr::null_mut(); 3];
        let mut uavs = [std::ptr::null_mut(); 6];
        for (s, slot) in batch.slots.iter().enumerate() {
            srvs[s] = if slot.from_plane {
                guard.planes[slot.src_component as usize].as_ref().ok_or("missing plane")?.srv
            } else {
                guard.bands[slot.src_component as usize][slot.src_level as usize]
                    .as_ref()
                    .ok_or("missing source band")?
                    .srv_layer0
            };
            let dst = guard.bands[slot.component as usize][slot.level as usize]
                .as_ref()
                .ok_or("missing dest band")?;
            uavs[s * 2] = dst.tex_uav;
            uavs[s * 2 + 1] = dst.coeff_uav;
        }
        out.push(DwtBind {
            srvs,
            uavs,
            cb: cb.buffer,
            gx: batch.groups_x,
            gy: batch.groups_y,
            gz: batch.groups_z,
        });
    }
    Ok(out)
}

unsafe fn bind_quant(guard: &FrameGuard, plan: &d3d11_plan::Submission, cbs: &[GpuBuf]) -> Result<Vec<QuantBind>, String> {
    let mut out = Vec::with_capacity(plan.quant.len());
    for (job, cb) in plan.quant.iter().zip(cbs.iter()) {
        let coeff = guard.bands[job.component as usize][job.level as usize]
            .as_ref()
            .ok_or("quant band missing")?;
        out.push(QuantBind {
            srv: coeff.coeff_srv,
            cb: cb.buffer,
            gx: job.groups_x,
            gy: job.groups_y,
            gz: job.groups_z,
        });
    }
    Ok(out)
}

unsafe fn run_pyramid(
    device: &Device,
    pyramid: &mut Pyramid,
    y: &[u8],
    u: &[u8],
    v: &[u8],
    mut blocks: Vec<BlockMeta>,
    mut recycle: Vec<u8>,
) -> Result<FrameBits, String> {
    let ctx = device.live.context;
    let planes_src = [y, u, v];
    let map_fn = pyramid.map_fn;
    let unmap_fn = pyramid.unmap_fn;
    let payload = pyramid.guard.bufs[2].buffer;
    let bitstream = pyramid.guard.bufs[4].buffer;
    for (i, (pw, ph)) in pyramid.plan.plane_size.iter().enumerate() {
        let tex = pyramid.guard.planes[i].as_ref().ok_or("missing plane")?.tex;
        upload_plane(ctx, map_fn, unmap_fn, tex, *pw, *ph, planes_src[i])?;
    }
    submit_gpu(ctx, device, pyramid)?;
    let copied = pyramid.copy_bits;
    let bit_bytes = readback_pack(ctx, pyramid.stage, payload, bitstream, copied, &mut recycle)?;
    // The first frames learn the packed size. Later frames copy that plus a page,
    // instead of the 128 KiB cap. Identical frames keep the same byte count.
    let next = ((bit_bytes as u32).saturating_add(4096) + 255) & !255;
    if next >= 256 && next < pyramid.copy_bits && bit_bytes < copied as usize {
        pyramid.copy_bits = next;
    }
    blocks.clear();
    blocks.push(BlockMeta { bytes: recycle });
    Ok(FrameBits {
        width: pyramid.width,
        height: pyramid.height,
        chroma_444: pyramid.chroma_444,
        blocks,
    })
}

unsafe fn submit_gpu(ctx: *mut c_void, device: &Device, pyramid: &Pyramid) -> Result<(), String> {
    let guard = &pyramid.guard;
    // One UAV unbind, then the next SRVs. Binding the SRVs while the
    // previous UAVs are still set is a hazard on the level that was just written.
    const NULL6: [*mut c_void; 6] = [std::ptr::null_mut(); 6];
    cs_set_shader(ctx, device.live.shaders[0]);
    cs_set_samplers(ctx, guard.sampler);
    for batch in &pyramid.dwt_bind {
        // Unbind the previous level before the next SRV. Leaving that UAV
        // bound produced an empty bitstream on the RTX 3060.
        cs_set_uavs(ctx, 0, &NULL6);
        cs_set_cb(ctx, batch.cb);
        cs_set_srvs(ctx, 3, &batch.srvs);
        cs_set_uavs(ctx, 0, &batch.uavs);
        dispatch(ctx, batch.gx, batch.gy, batch.gz);
    }
    cs_set_uavs(ctx, 0, &NULL6);
    cs_set_srvs(ctx, 3, &[std::ptr::null_mut(); 3]);
    cs_set_samplers(ctx, std::ptr::null_mut());
    // One-shot, env-gated. Confirms the dynamic-texture upload and the DWT
    // output. Off unless PWCMP_CHECK is set, so the timed path does not copy.
    if check_enabled() {
        probe_frame(device.live.device, ctx, pyramid);
    }

    // Quant rewrites every visited meta and stats slot. Only the payload
    // counters have to start at zero. frame.cpp clears 16 bytes here.
    clear_head(ctx, guard.bufs[2].buffer);
    cs_set_shader(ctx, device.live.shaders[1]);
    cs_set_uavs(ctx, 0, &[guard.bufs[0].uav, guard.bufs[1].uav, guard.bufs[2].uav]);
    for job in &pyramid.quant_bind {
        cs_set_cb(ctx, job.cb);
        cs_set_srvs(ctx, 0, &[job.srv]);
        dispatch(ctx, job.gx, job.gy, job.gz);
    }
    cs_set_srvs(ctx, 0, &[std::ptr::null_mut()]);
    cs_set_uavs(ctx, 0, &[std::ptr::null_mut(); 3]);

    cs_set_shader(ctx, device.live.shaders[2]);
    cs_set_srvs(
        ctx,
        0,
        &[guard.bufs[0].srv, guard.bufs[1].srv, guard.bufs[3].srv, guard.bufs[6].srv],
    );
    cs_set_uavs(ctx, 0, &[guard.bufs[4].uav, guard.bufs[5].uav, guard.bufs[2].uav]);
    dispatch(ctx, pyramid.plan.pack_groups_x, pyramid.plan.pack_groups_y, pyramid.plan.pack_groups_z);
    cs_set_srvs(ctx, 0, &[std::ptr::null_mut(); 4]);
    cs_set_uavs(ctx, 0, &[std::ptr::null_mut(); 3]);
    Ok(())
}

unsafe fn upload_plane(
    ctx: *mut c_void,
    map: MapFn,
    unmap: UnmapFn,
    tex: *mut c_void,
    width: u32,
    height: u32,
    samples: &[u8],
) -> Result<(), String> {
    let need = (width as usize).saturating_mul(height as usize);
    if samples.len() < need {
        return Err("plane is short".into());
    }
    let mut mapped = Mapped { data: std::ptr::null_mut(), row: 0, depth: 0 };
    let hr = map(ctx, tex, 0, MAP_WRITE_DISCARD, 0, &mut mapped);
    if hr < 0 || mapped.data.is_null() {
        return Err(format!("Map(plane) failed: hr={hr:#x}"));
    }
    let dst = mapped.data as *mut u8;
    let row_bytes = width as usize;
    let pitch = mapped.row as usize;
    if check_enabled() {
        eprintln!(
            "MAPGEOM ptr={:?} align={} pitch={pitch} row={row_bytes} h={height}",
            dst,
            (dst as usize) & 31
        );
    }
    if pitch == row_bytes {
        // DYNAMIC maps on the RTX 3060 are write-combined. A cached memcpy
        // stalls on those stores. Streaming stores match the DWT band hash of
        // memcpy (PWCMP_CHECK, both resolutions) and move the rows faster.
        copy_nt(samples.as_ptr(), dst, need);
    } else if pitch > row_bytes {
        copy_plane_rows_nt(samples.as_ptr(), dst, row_bytes, pitch, height as usize);
    } else {
        unmap(ctx, tex, 0);
        return Err(format!("Map(plane) row pitch {pitch} is shorter than {row_bytes}"));
    }
    unmap(ctx, tex, 0);
    Ok(())
}

fn check_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("PWCMP_CHECK").is_some())
}

/// Non-temporal copy. `_mm256_stream_si256` faults unless `dst` is 32-byte aligned,
/// so a short head uses ordinary stores. An sfence makes the stores visible.
pub(crate) unsafe fn copy_nt(src: *const u8, dst: *mut u8, n: usize) {
    #[cfg(target_arch = "x86_64")]
    if avx2_ok() {
        nt_copy_avx2(src, dst, n);
        std::arch::x86_64::_mm_sfence();
        return;
    }
    std::ptr::copy_nonoverlapping(src, dst, n);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn nt_copy_avx2(src: *const u8, dst: *mut u8, n: usize) {
    use std::arch::x86_64::{_mm256_loadu_si256, _mm256_stream_si256};
    let mut i = 0usize;
    let mis = (dst as usize) & 31;
    if mis != 0 && n > 0 {
        let head = (32 - mis).min(n);
        std::ptr::copy_nonoverlapping(src, dst, head);
        i = head;
    }
    while i + 128 <= n {
        let a = _mm256_loadu_si256(src.add(i) as *const _);
        let b = _mm256_loadu_si256(src.add(i + 32) as *const _);
        let c = _mm256_loadu_si256(src.add(i + 64) as *const _);
        let d = _mm256_loadu_si256(src.add(i + 96) as *const _);
        _mm256_stream_si256(dst.add(i) as *mut _, a);
        _mm256_stream_si256(dst.add(i + 32) as *mut _, b);
        _mm256_stream_si256(dst.add(i + 64) as *mut _, c);
        _mm256_stream_si256(dst.add(i + 96) as *mut _, d);
        i += 128;
    }
    while i + 32 <= n {
        let a = _mm256_loadu_si256(src.add(i) as *const _);
        _mm256_stream_si256(dst.add(i) as *mut _, a);
        i += 32;
    }
    if i < n {
        std::ptr::copy_nonoverlapping(src.add(i), dst.add(i), n - i);
    }
}

unsafe fn copy_plane_rows_nt(src: *const u8, dst: *mut u8, row: usize, pitch: usize, height: usize) {
    if row == 0 || height == 0 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if avx2_ok() {
        for y in 0..height {
            nt_copy_avx2(src.add(y * row), dst.add(y * pitch), row);
        }
        std::arch::x86_64::_mm_sfence();
        return;
    }
    for y in 0..height {
        std::ptr::copy_nonoverlapping(src.add(y * row), dst.add(y * pitch), row);
    }
}

static PROBES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// After the DWT, hash the coarsest luma band and try to read the Y plane back.
/// Two processes (memcpy vs streaming stores) must print the same BANDHASH when
/// the GPU saw the same samples. Runs at most twice per process.
unsafe fn probe_frame(device: *mut c_void, ctx: *mut c_void, pyramid: &Pyramid) {
    let n = PROBES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if n >= 2 {
        return;
    }
    let y = pyramid.guard.planes[0].as_ref();
    if let Some(plane) = y {
        let (w, h) = pyramid.plan.plane_size[0];
        probe_plane(device, ctx, plane.tex, w, h);
    }
    let bw = (pyramid.plan.aligned_w >> 5).max(0) as u32;
    let bh = (pyramid.plan.aligned_h >> 5).max(0) as u32;
    if bw > 0 && bh > 0 {
        if let Some(band) = pyramid.guard.bands[0][4].as_ref() {
            probe_band(device, ctx, band.tex, bw, bh);
        } else {
            eprintln!("BANDHASH missing level-4 luma");
        }
    }
}

unsafe fn probe_plane(device: *mut c_void, ctx: *mut c_void, src_tex: *mut c_void, width: u32, height: u32) {
    let desc = d3d11_plan::TextureBinding {
        width,
        height,
        format: FORMAT_R8_UNORM,
        array_size: 1,
        srv_dimension: 0,
        uav_dimension: 0,
        srv_array_size: 0,
        uav_array_size: 0,
    };
    let stage = match create_texture(device, &desc, USAGE_STAGING, 0, CPU_ACCESS_READ, 0) {
        Ok(tex) => tex,
        Err(err) => {
            eprintln!("PLANECHECK create {err}");
            return;
        }
    };
    type CopyResource = unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void);
    let copy: CopyResource = vslot(ctx, 47);
    copy(ctx, stage, src_tex);
    let mut mapped = Mapped { data: std::ptr::null_mut(), row: 0, depth: 0 };
    type Map = unsafe extern "system" fn(*mut c_void, *mut c_void, u32, u32, u32, *mut Mapped) -> i32;
    let map: Map = vslot(ctx, 14);
    let unmap: UnmapFn = vslot(ctx, 15);
    let hr = map(ctx, stage, 0, MAP_READ, 0, &mut mapped);
    if hr < 0 || mapped.data.is_null() {
        eprintln!("PLANECHECK map hr={hr:#x}");
        release(stage);
        return;
    }
    let pitch = mapped.row as usize;
    let row = width as usize;
    let mut mismatches = 0u32;
    let mut first = 0usize;
    if pitch >= row {
        let gpu = mapped.data as *const u8;
        for y in 0..height as usize {
            for x in 0..row {
                let g = *gpu.add(y * pitch + x);
                // The probe has no CPU source pointer here. A zero image means
                // CopyResource did not see the dynamic upload.
                if g == 0 {
                    mismatches += 1;
                    if mismatches == 1 {
                        first = y * row + x;
                    }
                }
            }
        }
    }
    let gpu0 = if mapped.data.is_null() {
        0
    } else {
        *(mapped.data as *const u8)
    };
    eprintln!(
        "PLANECHECK zeros={mismatches} first={first} gpu0={gpu0} pitch={pitch} {width}x{height} hr={hr:#x}"
    );
    unmap(ctx, stage, 0);
    release(stage);
}

unsafe fn probe_band(device: *mut c_void, ctx: *mut c_void, src_tex: *mut c_void, width: u32, height: u32) {
    let desc = d3d11_plan::TextureBinding {
        width,
        height,
        format: FORMAT_R32_FLOAT,
        array_size: 4,
        srv_dimension: 0,
        uav_dimension: 0,
        srv_array_size: 0,
        uav_array_size: 0,
    };
    let stage = match create_texture(device, &desc, USAGE_STAGING, 0, CPU_ACCESS_READ, 0) {
        Ok(tex) => tex,
        Err(err) => {
            eprintln!("BANDHASH create {err}");
            return;
        }
    };
    type CopyResource = unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void);
    let copy: CopyResource = vslot(ctx, 47);
    copy(ctx, stage, src_tex);
    let mut mapped = Mapped { data: std::ptr::null_mut(), row: 0, depth: 0 };
    type Map = unsafe extern "system" fn(*mut c_void, *mut c_void, u32, u32, u32, *mut Mapped) -> i32;
    let map: Map = vslot(ctx, 14);
    let unmap: UnmapFn = vslot(ctx, 15);
    let hr = map(ctx, stage, 0, MAP_READ, 0, &mut mapped);
    if hr < 0 || mapped.data.is_null() {
        eprintln!("BANDHASH map hr={hr:#x}");
        release(stage);
        return;
    }
    let pitch = mapped.row as usize;
    let slice_pitch = if mapped.depth == 0 { pitch.saturating_mul(height as usize) } else { mapped.depth as usize };
    let row_bytes = (width as usize).saturating_mul(4);
    let mut h = 0xcbf29ce484222325u64;
    let base = mapped.data as *const u8;
    if pitch >= row_bytes && slice_pitch >= pitch.saturating_mul(height as usize) {
        for slice in 0..4usize {
            for y in 0..height as usize {
                let rowp = base.add(slice * slice_pitch + y * pitch);
                let bytes = std::slice::from_raw_parts(rowp, row_bytes);
                for &b in bytes {
                    h ^= b as u64;
                    h = h.wrapping_mul(0x100000001b3);
                }
            }
        }
    }
    let f0 = if row_bytes >= 4 {
        f32::from_le_bytes([*base, *base.add(1), *base.add(2), *base.add(3)])
    } else {
        0.0
    };
    eprintln!("BANDHASH {h:016x} f0={f0} {width}x{height} pitch={pitch} depth={} hr={hr:#x}", mapped.depth);
    unmap(ctx, stage, 0);
    release(stage);
}

/// Copy tightly packed rows into a texture map whose pitch is wider.
/// The AVX2 loop stays in the instruction cache; a per-row `memcpy` call does not.
pub(crate) unsafe fn avx_plane(src: *const u8, dst: *mut u8, n: usize) {
    #[cfg(target_arch = "x86_64")]
    if avx2_ok() {
        avx_plane_store(src, dst, n);
        return;
    }
    std::ptr::copy_nonoverlapping(src, dst, n);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx_plane_store(src: *const u8, dst: *mut u8, n: usize) {
    use std::arch::x86_64::{_mm256_loadu_si256, _mm256_storeu_si256};
    let mut i = 0usize;
    while i + 128 <= n {
        let a = _mm256_loadu_si256(src.add(i) as *const _);
        let b = _mm256_loadu_si256(src.add(i + 32) as *const _);
        let c = _mm256_loadu_si256(src.add(i + 64) as *const _);
        let d = _mm256_loadu_si256(src.add(i + 96) as *const _);
        _mm256_storeu_si256(dst.add(i) as *mut _, a);
        _mm256_storeu_si256(dst.add(i + 32) as *mut _, b);
        _mm256_storeu_si256(dst.add(i + 64) as *mut _, c);
        _mm256_storeu_si256(dst.add(i + 96) as *mut _, d);
        i += 128;
    }
    while i + 32 <= n {
        let a = _mm256_loadu_si256(src.add(i) as *const _);
        _mm256_storeu_si256(dst.add(i) as *mut _, a);
        i += 32;
    }
    if i < n {
        std::ptr::copy_nonoverlapping(src.add(i), dst.add(i), n - i);
    }
}

pub(crate) unsafe fn copy_plane_rows(src: *const u8, dst: *mut u8, row: usize, pitch: usize, height: usize) {
    if row == 0 || height == 0 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if avx2_ok() {
        copy_pitched_avx2(src, dst, row, pitch, height);
        return;
    }
    for y in 0..height {
        std::ptr::copy_nonoverlapping(src.add(y * row), dst.add(y * pitch), row);
    }
}

fn avx2_ok() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn copy_pitched_avx2(src: *const u8, dst: *mut u8, row: usize, pitch: usize, height: usize) {
    use std::arch::x86_64::{_mm256_loadu_si256, _mm256_storeu_si256};
    let mut y = 0usize;
    while y < height {
        let s = src.add(y * row);
        let d = dst.add(y * pitch);
        let mut x = 0usize;
        while x + 128 <= row {
            let a = _mm256_loadu_si256(s.add(x) as *const _);
            let b = _mm256_loadu_si256(s.add(x + 32) as *const _);
            let c = _mm256_loadu_si256(s.add(x + 64) as *const _);
            let e = _mm256_loadu_si256(s.add(x + 96) as *const _);
            _mm256_storeu_si256(d.add(x) as *mut _, a);
            _mm256_storeu_si256(d.add(x + 32) as *mut _, b);
            _mm256_storeu_si256(d.add(x + 64) as *mut _, c);
            _mm256_storeu_si256(d.add(x + 96) as *mut _, e);
            x += 128;
        }
        while x + 32 <= row {
            let a = _mm256_loadu_si256(s.add(x) as *const _);
            _mm256_storeu_si256(d.add(x) as *mut _, a);
            x += 32;
        }
        if x < row {
            std::ptr::copy_nonoverlapping(s.add(x), d.add(x), row - x);
        }
        y += 1;
    }
}

struct PlaneGpu {
    tex: *mut c_void,
    srv: *mut c_void,
}

struct BandGpu {
    tex: *mut c_void,
    tex_uav: *mut c_void,
    srv: *mut c_void,
    srv_layer0: *mut c_void,
    coeff_uav: *mut c_void,
    coeff_srv: *mut c_void,
    coeff: *mut c_void,
}

struct GpuBuf {
    buffer: *mut c_void,
    uav: *mut c_void,
    srv: *mut c_void,
    bytes: usize,
    staging: *mut c_void,
}

fn destroy_buf(b: GpuBuf) {
    unsafe {
        release(b.uav);
        release(b.srv);
        release(b.staging);
        release(b.buffer);
    }
}

/// Frees one constant buffer at the end of the dispatch that wrote it.
struct BufDrop(Option<GpuBuf>);

impl BufDrop {
    fn buffer(&self) -> *mut c_void {
        self.0.as_ref().map(|b| b.buffer).unwrap_or(std::ptr::null_mut())
    }
}

impl Drop for BufDrop {
    fn drop(&mut self) {
        if let Some(buf) = self.0.take() {
            destroy_buf(buf);
        }
    }
}

struct FrameGuard {
    ctx: *mut c_void,
    planes: [Option<PlaneGpu>; 3],
    bands: [[Option<BandGpu>; 5]; 3],
    sampler: *mut c_void,
    bufs: Vec<GpuBuf>,
}

impl Drop for FrameGuard {
    fn drop(&mut self) {
        unsafe {
            if !self.ctx.is_null() {
                cs_set_shader(self.ctx, std::ptr::null_mut());
                cs_set_samplers(self.ctx, std::ptr::null_mut());
                cs_set_cb(self.ctx, std::ptr::null_mut());
                cs_set_uavs(self.ctx, 0, &[std::ptr::null_mut(); 6]);
                cs_set_srvs(self.ctx, 0, &[std::ptr::null_mut(); 4]);
                cs_set_srvs(self.ctx, 3, &[std::ptr::null_mut(); 3]);
            }
            for plane in self.planes.iter_mut() {
                if let Some(plane) = plane.take() {
                    release(plane.srv);
                    release(plane.tex);
                }
            }
            for row in self.bands.iter_mut() {
                for band in row.iter_mut() {
                    if let Some(band) = band.take() {
                        destroy_band(&band);
                    }
                }
            }
            release(self.sampler);
            for buf in self.bufs.drain(..) {
                destroy_buf(buf);
            }
        }
    }
}

const USAGE_DEFAULT: u32 = 0;
const USAGE_DYNAMIC: u32 = 2;
const USAGE_STAGING: u32 = 3;
const BIND_CONSTANT_BUFFER: u32 = 4;
const BIND_SHADER_RESOURCE: u32 = 8;
const BIND_UNORDERED_ACCESS: u32 = 0x80;
const CPU_ACCESS_WRITE: u32 = 0x10000;
const CPU_ACCESS_READ: u32 = 0x20000;
const MISC_RAW: u32 = 0x20;
const MAP_READ: u32 = 1;
const MAP_WRITE_DISCARD: u32 = 4;
const SRV_DIM_BUFFER: u32 = 1;
const ADDRESS_MIRROR: u32 = 2;

#[repr(C)]
struct TexDesc {
    width: u32,
    height: u32,
    mips: u32,
    array: u32,
    format: u32,
    sample_count: u32,
    sample_quality: u32,
    usage: u32,
    bind: u32,
    cpu: u32,
    misc: u32,
}

#[repr(C)]
struct BufDesc {
    byte_width: u32,
    usage: u32,
    bind: u32,
    cpu: u32,
    misc: u32,
    stride: u32,
}

/// 24 bytes. The view union is 16 bytes. `d3d11.h` `D3D11_SHADER_RESOURCE_VIEW_DESC`.
#[repr(C)]
struct SrvDesc {
    format: u32,
    dimension: u32,
    a: u32,
    b: u32,
    c: u32,
    d: u32,
}

/// 20 bytes. `D3D11_UNORDERED_ACCESS_VIEW_DESC`.
#[repr(C)]
struct UavDesc {
    format: u32,
    dimension: u32,
    a: u32,
    b: u32,
    c: u32,
}

#[repr(C)]
struct Mapped {
    data: *mut c_void,
    row: u32,
    depth: u32,
}

unsafe fn create_plane(
    device: *mut c_void,
    ctx: *mut c_void,
    desc: &d3d11_plan::TextureBinding,
    samples: &[u8],
) -> Result<PlaneGpu, String> {
    // R8_UNORM, one array slice, DYNAMIC. dwt.cpp:401-418.
    if desc.format != FORMAT_R8_UNORM || desc.array_size != 1 || desc.srv_dimension != SRV_DIM_TEXTURE2DARRAY {
        return Err("plane binding is not an R8 Texture2DArray".into());
    }
    let tex = create_texture(device, desc, USAGE_DYNAMIC, BIND_SHADER_RESOURCE, CPU_ACCESS_WRITE, 0)?;
    let srv = match make_srv_array(device, tex, desc.format, desc.srv_dimension, 0, desc.srv_array_size) {
        Ok(view) => view,
        Err(err) => {
            release(tex);
            return Err(err);
        }
    };
    let need = (desc.width as usize).saturating_mul(desc.height as usize);
    if samples.len() < need {
        release(srv);
        release(tex);
        return Err("plane is short".into());
    }
    type MapFn = unsafe extern "system" fn(*mut c_void, *mut c_void, u32, u32, u32, *mut Mapped) -> i32;
    type UnmapFn = unsafe extern "system" fn(*mut c_void, *mut c_void, u32);
    let map: MapFn = vslot(ctx, 14);
    let unmap: UnmapFn = vslot(ctx, 15);
    let mut mapped = Mapped {
        data: std::ptr::null_mut(),
        row: 0,
        depth: 0,
    };
    let hr = map(ctx, tex, 0, MAP_WRITE_DISCARD, 0, &mut mapped);
    if hr < 0 || mapped.data.is_null() {
        release(srv);
        release(tex);
        return Err(format!("Map(plane) failed: hr={hr:#x}"));
    }
    let dst = mapped.data as *mut u8;
    let row_bytes = desc.width as usize;
    if mapped.row as usize == row_bytes {
        std::ptr::copy_nonoverlapping(samples.as_ptr(), dst, need);
    } else {
        for y in 0..desc.height as usize {
            std::ptr::copy_nonoverlapping(
                samples.as_ptr().add(y * row_bytes),
                dst.add(y * mapped.row as usize),
                row_bytes,
            );
        }
    }
    unmap(ctx, tex, 0);
    Ok(PlaneGpu { tex, srv })
}

unsafe fn create_band(
    device: *mut c_void,
    ctx: *mut c_void,
    desc: &d3d11_plan::TextureBinding,
) -> Result<BandGpu, String> {
    // Four-layer R32_FLOAT. dwt.cpp make_band.
    if desc.format != FORMAT_R32_FLOAT
        || desc.array_size != 4
        || desc.uav_dimension != UAV_DIM_TEXTURE2DARRAY
        || desc.srv_dimension != SRV_DIM_TEXTURE2DARRAY
    {
        return Err("band binding is not a 4-layer R32_FLOAT Texture2DArray".into());
    }
    let tex = create_texture(
        device,
        desc,
        USAGE_DEFAULT,
        BIND_UNORDERED_ACCESS | BIND_SHADER_RESOURCE,
        0,
        0,
    )?;
    let tex_uav = match make_uav_array(device, tex, desc.format, desc.uav_dimension, desc.uav_array_size) {
        Ok(view) => view,
        Err(err) => {
            release(tex);
            return Err(err);
        }
    };
    let srv_full = match make_srv_array(device, tex, desc.format, desc.srv_dimension, 0, desc.srv_array_size) {
        Ok(view) => view,
        Err(err) => {
            release(tex_uav);
            release(tex);
            return Err(err);
        }
    };
    let srv_layer0 = match make_srv_array(device, tex, desc.format, desc.srv_dimension, 0, 1) {
        Ok(view) => view,
        Err(err) => {
            release(srv_full);
            release(tex_uav);
            release(tex);
            return Err(err);
        }
    };
    let pitch = ((desc.width + 31) & !31) as usize;
    let pitch_h = ((desc.height + 31) & !31) as usize;
    let elems = 4 * pitch * pitch_h;
    let coeff = match create_buffer(device, elems * 4, USAGE_DEFAULT, BIND_UNORDERED_ACCESS | BIND_SHADER_RESOURCE, 0, MISC_RAW) {
        Ok(buf) => buf,
        Err(err) => {
            release(srv_layer0);
            release(srv_full);
            release(tex_uav);
            release(tex);
            return Err(err);
        }
    };
    // RWBuffer<float>: typed R32_FLOAT, not a raw view. dwt.cpp:164-168.
    let coeff_uav = match make_uav_buffer(device, coeff, FORMAT_R32_FLOAT, elems as u32, 0) {
        Ok(view) => view,
        Err(err) => {
            release(coeff);
            release(srv_layer0);
            release(srv_full);
            release(tex_uav);
            release(tex);
            return Err(err);
        }
    };
    // Quant's t0 is the raw coefficient SRV. dwt.cpp:171-176, dwt_get_band_srv.
    let coeff_srv = match make_srv_raw(device, coeff, elems as u32) {
        Ok(view) => view,
        Err(err) => {
            release(coeff_uav);
            release(coeff);
            release(srv_layer0);
            release(srv_full);
            release(tex_uav);
            release(tex);
            return Err(err);
        }
    };
    let zero = [0f32; 4];
    type ClearF = unsafe extern "system" fn(*mut c_void, *mut c_void, *const f32);
    let clear: ClearF = vslot(ctx, 52);
    clear(ctx, tex_uav, zero.as_ptr());
    clear(ctx, coeff_uav, zero.as_ptr());
    Ok(BandGpu {
        tex,
        tex_uav,
        srv: srv_full,
        srv_layer0,
        coeff_uav,
        coeff_srv,
        coeff,
    })
}

unsafe fn create_sampler(device: *mut c_void) -> Result<*mut c_void, String> {
    // dwt.cpp:371-375. Comparison stays 0, matching that zeroed desc.
    #[repr(C)]
    struct SamplerDesc {
        filter: u32,
        address_u: u32,
        address_v: u32,
        address_w: u32,
        mip_bias: f32,
        max_aniso: u32,
        comparison: u32,
        border: [f32; 4],
        min_lod: f32,
        max_lod: f32,
    }
    let desc = SamplerDesc {
        filter: 0,
        address_u: ADDRESS_MIRROR,
        address_v: ADDRESS_MIRROR,
        address_w: ADDRESS_MIRROR,
        mip_bias: 0.0,
        max_aniso: 0,
        comparison: 0,
        border: [0.0; 4],
        min_lod: 0.0,
        max_lod: f32::MAX,
    };
    type Create = unsafe extern "system" fn(*mut c_void, *const SamplerDesc, *mut *mut c_void) -> i32;
    let create: Create = vslot(device, 23);
    let mut sampler = std::ptr::null_mut();
    let hr = create(device, &desc, &mut sampler);
    if hr < 0 || sampler.is_null() {
        return Err(format!("CreateSamplerState failed: hr={hr:#x}"));
    }
    Ok(sampler)
}

unsafe fn create_uint_buffer(device: *mut c_void, bytes: usize, want_uav: bool, want_srv: bool) -> Result<GpuBuf, String> {
    let bytes = align16(bytes);
    let mut bind = 0u32;
    if want_uav {
        bind |= BIND_UNORDERED_ACCESS;
    }
    if want_srv {
        bind |= BIND_SHADER_RESOURCE;
    }
    // Typed R32_UINT, the views quant.cpp and frame.cpp create. Not raw.
    let buffer = create_buffer(device, bytes, USAGE_DEFAULT, bind, 0, 0)?;
    let elems = (bytes / 4) as u32;
    let uav = if want_uav {
        match make_uav_buffer(device, buffer, FORMAT_R32_UINT, elems, 0) {
            Ok(view) => view,
            Err(err) => {
                release(buffer);
                return Err(err);
            }
        }
    } else {
        std::ptr::null_mut()
    };
    let srv = if want_srv {
        match make_srv_typed(device, buffer, FORMAT_R32_UINT, SRV_DIM_BUFFER, elems) {
            Ok(view) => view,
            Err(err) => {
                release(uav);
                release(buffer);
                return Err(err);
            }
        }
    } else {
        std::ptr::null_mut()
    };
    let staging = match create_staging(device, bytes) {
        Ok(buf) => buf,
        Err(err) => {
            release(srv);
            release(uav);
            release(buffer);
            return Err(err);
        }
    };
    Ok(GpuBuf {
        buffer,
        uav,
        srv,
        bytes,
        staging,
    })
}

unsafe fn create_sint_buffer(device: *mut c_void, bytes: usize) -> Result<GpuBuf, String> {
    let bytes = align16(bytes);
    let elems = (bytes / 4) as u32;
    // analyze.cpp quant buffer: R32_SINT UAV and, for pack, an R32_SINT SRV.
    let buffer = create_buffer(
        device,
        bytes,
        USAGE_DEFAULT,
        BIND_UNORDERED_ACCESS | BIND_SHADER_RESOURCE,
        0,
        0,
    )?;
    let uav = match make_uav_buffer(device, buffer, FORMAT_R32_SINT, elems, 0) {
        Ok(view) => view,
        Err(err) => {
            release(buffer);
            return Err(err);
        }
    };
    let srv = match make_srv_typed(device, buffer, FORMAT_R32_SINT, SRV_DIM_BUFFER, elems) {
        Ok(view) => view,
        Err(err) => {
            release(uav);
            release(buffer);
            return Err(err);
        }
    };
    Ok(GpuBuf {
        buffer,
        uav,
        srv,
        bytes,
        staging: std::ptr::null_mut(),
    })
}

unsafe fn create_raw_srv_buffer(device: *mut c_void, bytes: &[u8]) -> Result<GpuBuf, String> {
    let size = align16(bytes.len());
    // frame.cpp pack jobs: raw BUFFEREX SRV. The bytes are UpdateSubresource'd.
    let buffer = create_buffer(device, size, USAGE_DEFAULT, BIND_SHADER_RESOURCE, 0, MISC_RAW)?;
    let srv = match make_srv_raw(device, buffer, (size / 4) as u32) {
        Ok(view) => view,
        Err(err) => {
            release(buffer);
            return Err(err);
        }
    };
    Ok(GpuBuf {
        buffer,
        uav: std::ptr::null_mut(),
        srv,
        bytes: size,
        staging: std::ptr::null_mut(),
    })
}

unsafe fn create_cb_bytes(device: *mut c_void, bytes: &[u8]) -> Result<GpuBuf, String> {
    let size = align16(bytes.len());
    let buffer = create_buffer(device, size, USAGE_DEFAULT, BIND_CONSTANT_BUFFER, 0, 0)?;
    Ok(GpuBuf {
        buffer,
        uav: std::ptr::null_mut(),
        srv: std::ptr::null_mut(),
        bytes: size,
        staging: std::ptr::null_mut(),
    })
}

fn align16(bytes: usize) -> usize {
    (bytes.max(16) + 15) & !15
}

unsafe fn create_texture(
    device: *mut c_void,
    desc: &d3d11_plan::TextureBinding,
    usage: u32,
    bind: u32,
    cpu: u32,
    misc: u32,
) -> Result<*mut c_void, String> {
    let td = TexDesc {
        width: desc.width,
        height: desc.height,
        mips: 1,
        array: desc.array_size,
        format: desc.format,
        sample_count: 1,
        sample_quality: 0,
        usage,
        bind,
        cpu,
        misc,
    };
    type CreateTexture2D = unsafe extern "system" fn(*mut c_void, *const TexDesc, *const c_void, *mut *mut c_void) -> i32;
    let create: CreateTexture2D = vslot(device, 5);
    let mut tex = std::ptr::null_mut();
    let hr = create(device, &td, std::ptr::null(), &mut tex);
    if hr < 0 || tex.is_null() {
        return Err(format!("CreateTexture2D failed: hr={hr:#x}"));
    }
    Ok(tex)
}

unsafe fn create_buffer(
    device: *mut c_void,
    bytes: usize,
    usage: u32,
    bind: u32,
    cpu: u32,
    misc: u32,
) -> Result<*mut c_void, String> {
    let desc = BufDesc {
        byte_width: bytes as u32,
        usage,
        bind,
        cpu,
        misc,
        stride: 0,
    };
    type CreateBuffer = unsafe extern "system" fn(*mut c_void, *const BufDesc, *const c_void, *mut *mut c_void) -> i32;
    let create: CreateBuffer = vslot(device, 3);
    let mut buffer = std::ptr::null_mut();
    let hr = create(device, &desc, std::ptr::null(), &mut buffer);
    if hr < 0 || buffer.is_null() {
        return Err(format!("CreateBuffer failed: hr={hr:#x}"));
    }
    Ok(buffer)
}

unsafe fn make_srv_array(
    device: *mut c_void,
    resource: *mut c_void,
    format: u32,
    dimension: u32,
    first: u32,
    array_size: u32,
) -> Result<*mut c_void, String> {
    // Texture2DArray: MostDetailedMip, MipLevels, FirstArraySlice, ArraySize.
    let desc = SrvDesc {
        format,
        dimension,
        a: 0,
        b: 1,
        c: first,
        d: array_size,
    };
    make_srv(device, resource, &desc)
}

unsafe fn make_srv_typed(
    device: *mut c_void,
    resource: *mut c_void,
    format: u32,
    dimension: u32,
    elems: u32,
) -> Result<*mut c_void, String> {
    // BUFFER SRV: FirstElement, NumElements, unused union pad.
    let desc = SrvDesc {
        format,
        dimension,
        a: 0,
        b: elems,
        c: 0,
        d: 0,
    };
    make_srv(device, resource, &desc)
}

unsafe fn make_srv_raw(device: *mut c_void, resource: *mut c_void, elems: u32) -> Result<*mut c_void, String> {
    // BUFFEREX: FirstElement, NumElements, Flags = RAW.
    let desc = SrvDesc {
        format: FORMAT_R32_TYPELESS,
        dimension: SRV_DIM_BUFFEREX,
        a: 0,
        b: elems,
        c: 1,
        d: 0,
    };
    make_srv(device, resource, &desc)
}

unsafe fn make_srv(device: *mut c_void, resource: *mut c_void, desc: &SrvDesc) -> Result<*mut c_void, String> {
    type CreateSrv = unsafe extern "system" fn(*mut c_void, *mut c_void, *const SrvDesc, *mut *mut c_void) -> i32;
    let mk: CreateSrv = vslot(device, 7);
    let mut view = std::ptr::null_mut();
    let hr = mk(device, resource, desc, &mut view);
    if hr < 0 || view.is_null() {
        return Err(format!("CreateShaderResourceView failed: hr={hr:#x}"));
    }
    Ok(view)
}

unsafe fn make_uav_array(
    device: *mut c_void,
    resource: *mut c_void,
    format: u32,
    dimension: u32,
    array_size: u32,
) -> Result<*mut c_void, String> {
    // Texture2DArray UAV: MipSlice, FirstArraySlice, ArraySize.
    let desc = UavDesc {
        format,
        dimension,
        a: 0,
        b: 0,
        c: array_size,
    };
    make_uav(device, resource, &desc)
}

unsafe fn make_uav_buffer(
    device: *mut c_void,
    resource: *mut c_void,
    format: u32,
    elems: u32,
    flags: u32,
) -> Result<*mut c_void, String> {
    let desc = UavDesc {
        format,
        dimension: UAV_DIM_BUFFER,
        a: 0,
        b: elems,
        c: flags,
    };
    make_uav(device, resource, &desc)
}

unsafe fn make_uav(device: *mut c_void, resource: *mut c_void, desc: &UavDesc) -> Result<*mut c_void, String> {
    type CreateUav = unsafe extern "system" fn(*mut c_void, *mut c_void, *const UavDesc, *mut *mut c_void) -> i32;
    let mk: CreateUav = vslot(device, 8);
    let mut view = std::ptr::null_mut();
    let hr = mk(device, resource, desc, &mut view);
    if hr < 0 || view.is_null() {
        return Err(format!("CreateUnorderedAccessView failed: hr={hr:#x}"));
    }
    Ok(view)
}

unsafe fn clear_head(ctx: *mut c_void, resource: *mut c_void) {
    #[repr(C)]
    struct Box16 {
        left: u32,
        top: u32,
        front: u32,
        right: u32,
        bottom: u32,
        back: u32,
    }
    let zeros = [0u8; 16];
    let region = Box16 { left: 0, top: 0, front: 0, right: 16, bottom: 1, back: 1 };
    type Update = unsafe extern "system" fn(*mut c_void, *mut c_void, u32, *const Box16, *const c_void, u32, u32);
    let update: Update = vslot(ctx, 48);
    update(ctx, resource, 0, &region, zeros.as_ptr() as *const c_void, 0, 0);
}

unsafe fn update_subresource(ctx: *mut c_void, resource: *mut c_void, bytes: &[u8]) {
    update_buffer(ctx, resource, bytes);
}

/// Whole DEFAULT resource. `bytes` must be the resource byte width: a null box
/// reads that many bytes (`dwt.cpp` constant buffers). A short slice faults.
unsafe fn update_whole(ctx: *mut c_void, resource: *mut c_void, bytes: &[u8]) {
    type Update = unsafe extern "system" fn(*mut c_void, *mut c_void, u32, *const c_void, *const c_void, u32, u32);
    let update: Update = vslot(ctx, 48);
    update(ctx, resource, 0, std::ptr::null(), bytes.as_ptr() as *const c_void, 0, 0);
}

/// Byte range of a DEFAULT buffer. `right` is the end byte. Pitches stay 0;
/// nonzero pitches are for textures and make this call copy nothing.
/// `frame.cpp` clears the payload head the same way.
unsafe fn update_buffer(ctx: *mut c_void, resource: *mut c_void, bytes: &[u8]) {
    #[repr(C)]
    struct ByteBox {
        left: u32,
        top: u32,
        front: u32,
        right: u32,
        bottom: u32,
        back: u32,
    }
    let region = ByteBox {
        left: 0,
        top: 0,
        front: 0,
        right: bytes.len() as u32,
        bottom: 1,
        back: 1,
    };
    type Update = unsafe extern "system" fn(*mut c_void, *mut c_void, u32, *const ByteBox, *const c_void, u32, u32);
    let update: Update = vslot(ctx, 48);
    update(ctx, resource, 0, &region, bytes.as_ptr() as *const c_void, 0, 0);
}

unsafe fn upload_zeros(ctx: *mut c_void, buf: &GpuBuf) {
    let zeros = vec![0u8; buf.bytes];
    update_subresource(ctx, buf.buffer, &zeros);
}

unsafe fn cs_set_srvs(ctx: *mut c_void, start: u32, views: &[*mut c_void]) {
    type SetSrvs = unsafe extern "system" fn(*mut c_void, u32, u32, *const *mut c_void);
    let set: SetSrvs = vslot(ctx, 67);
    set(ctx, start, views.len() as u32, views.as_ptr());
}

unsafe fn cs_set_uavs(ctx: *mut c_void, start: u32, views: &[*mut c_void]) {
    type SetUavs = unsafe extern "system" fn(*mut c_void, u32, u32, *const *mut c_void, *const u32);
    let set: SetUavs = vslot(ctx, 68);
    // -1 keeps an append offset. A null count array leaves the HD 6670 views unbound.
    let counts = vec![u32::MAX; views.len()];
    set(ctx, start, views.len() as u32, views.as_ptr(), counts.as_ptr());
}

unsafe fn cs_set_samplers(ctx: *mut c_void, sampler: *mut c_void) {
    type Set = unsafe extern "system" fn(*mut c_void, u32, u32, *const *mut c_void);
    let set: Set = vslot(ctx, 70);
    set(ctx, 0, 1, &sampler);
}

fn destroy_band(band: &BandGpu) {
    unsafe {
        release(band.coeff_srv);
        release(band.coeff_uav);
        release(band.coeff);
        release(band.srv_layer0);
        release(band.srv);
        release(band.tex_uav);
        release(band.tex);
    }
}

unsafe fn create_upload(device: *mut c_void, bytes: usize) -> Result<*mut c_void, String> {
    #[repr(C)]
    struct BufDesc {
        byte_width: u32,
        usage: u32,
        bind: u32,
        cpu: u32,
        misc: u32,
        stride: u32,
    }
    let desc = BufDesc {
        byte_width: bytes as u32,
        usage: USAGE_STAGING,
        bind: 0,
        cpu: CPU_ACCESS_WRITE,
        misc: 0,
        stride: 0,
    };
    type CreateBuffer = unsafe extern "system" fn(*mut c_void, *const BufDesc, *const c_void, *mut *mut c_void) -> i32;
    let create: CreateBuffer = vslot(device, 3);
    let mut buffer = std::ptr::null_mut();
    let hr = create(device, &desc, std::ptr::null(), &mut buffer);
    if hr < 0 || buffer.is_null() {
        return Err(format!("CreateBuffer(upload) failed: hr={hr:#x}"));
    }
    Ok(buffer)
}

unsafe fn create_staging(device: *mut c_void, bytes: usize) -> Result<*mut c_void, String> {
    #[repr(C)]
    struct BufDesc {
        byte_width: u32,
        usage: u32,
        bind: u32,
        cpu: u32,
        misc: u32,
        stride: u32,
    }
    let desc = BufDesc {
        byte_width: bytes as u32,
        usage: USAGE_STAGING,
        bind: 0,
        cpu: CPU_ACCESS_READ,
        misc: 0,
        stride: 0,
    };
    type CreateBuffer = unsafe extern "system" fn(*mut c_void, *const BufDesc, *const c_void, *mut *mut c_void) -> i32;
    let create: CreateBuffer = vslot(device, 3);
    let mut buffer = std::ptr::null_mut();
    let hr = create(device, &desc, std::ptr::null(), &mut buffer);
    if hr < 0 || buffer.is_null() {
        return Err(format!("CreateBuffer(staging) failed: hr={hr:#x}"));
    }
    Ok(buffer)
}

unsafe fn cs_set_shader(ctx: *mut c_void, shader: *mut c_void) {
    type Set = unsafe extern "system" fn(*mut c_void, *mut c_void, *const *mut c_void, u32);
    let set: Set = vslot(ctx, 69);
    set(ctx, shader, std::ptr::null(), 0);
}

unsafe fn cs_set_cb(ctx: *mut c_void, cb: *mut c_void) {
    type Set = unsafe extern "system" fn(*mut c_void, u32, u32, *const *mut c_void);
    let set: Set = vslot(ctx, 71);
    set(ctx, 0, 1, &cb);
}

unsafe fn dispatch(ctx: *mut c_void, x: u32, y: u32, z: u32) {
    type Dispatch = unsafe extern "system" fn(*mut c_void, u32, u32, u32);
    let go: Dispatch = vslot(ctx, 41);
    go(ctx, x, y, z);
}

unsafe fn copy_pack_to_stage(
    ctx: *mut c_void,
    stage: *mut c_void,
    payload: *mut c_void,
    bitstream: *mut c_void,
    copy_bits: u32,
) {
    #[repr(C)]
    struct BoxRegion {
        left: u32,
        top: u32,
        front: u32,
        right: u32,
        bottom: u32,
        back: u32,
    }
    type CopyBox = unsafe extern "system" fn(
        *mut c_void,
        *mut c_void,
        u32,
        u32,
        u32,
        u32,
        *mut c_void,
        u32,
        *const BoxRegion,
    );
    // ID3D11DeviceContext::CopySubresourceRegion sits just before CopyResource.
    let copy: CopyBox = vslot(ctx, 46);
    let head = BoxRegion { left: 0, top: 0, front: 0, right: 16, bottom: 1, back: 1 };
    copy(ctx, stage, 0, 0, 0, 0, payload, 0, &head);
    if copy_bits > 0 {
        let bits = BoxRegion { left: 0, top: 0, front: 0, right: copy_bits, bottom: 1, back: 1 };
        copy(ctx, stage, 0, 16, 0, 0, bitstream, 0, &bits);
    }
}

unsafe fn readback_pack(
    ctx: *mut c_void,
    stage: *mut c_void,
    payload: *mut c_void,
    bitstream: *mut c_void,
    copy_bits: u32,
    bits_buf: &mut Vec<u8>,
) -> Result<usize, String> {
    copy_pack_to_stage(ctx, stage, payload, bitstream, copy_bits);
    #[repr(C)]
    struct Mapped {
        data: *mut c_void,
        row: u32,
        depth: u32,
    }
    type Map = unsafe extern "system" fn(*mut c_void, *mut c_void, u32, u32, u32, *mut Mapped) -> i32;
    type Unmap = unsafe extern "system" fn(*mut c_void, *mut c_void, u32);
    let map: Map = vslot(ctx, 14);
    let unmap: Unmap = vslot(ctx, 15);
    let mut mapped = Mapped { data: std::ptr::null_mut(), row: 0, depth: 0 };
    let hr = map(ctx, stage, 0, MAP_READ, 0, &mut mapped);
    if hr < 0 || mapped.data.is_null() {
        return Err(format!("Map(staging) failed: hr={hr:#x}"));
    }
    let total = 16 + copy_bits as usize;
    let src = std::slice::from_raw_parts(mapped.data as *const u8, total);
    let words = u32::from_le_bytes([src[4], src[5], src[6], src[7]]) as usize;
    let mut bit_bytes = words.saturating_mul(4);
    if bit_bytes > copy_bits as usize {
        bit_bytes = copy_bits as usize;
    }
    if bit_bytes == 0 {
        unmap(ctx, stage, 0);
        return Err("GPU pack wrote no block packets".into());
    }
    let full = bit_bytes;
    if bits_buf.capacity() < bit_bytes {
        bits_buf.reserve(bit_bytes - bits_buf.capacity());
    }
    unsafe {
        bits_buf.set_len(bit_bytes);
    }
    std::ptr::copy_nonoverlapping(src.as_ptr().add(16), bits_buf.as_mut_ptr(), bit_bytes);
    unmap(ctx, stage, 0);
    Ok(full)
}

unsafe fn create_cs_bytes(device: *mut c_void, name: &str, bytes: &[u8]) -> Result<*mut c_void, String> {
    if bytes.len() < 4 || &bytes[..4] != b"DXBC" {
        return Err(format!("{name} is not DXBC"));
    }
    type CreateCs = unsafe extern "system" fn(*mut c_void, *const c_void, usize, *mut c_void, *mut *mut c_void) -> i32;
    let mk: CreateCs = vslot(device, 18);
    let mut shader = std::ptr::null_mut();
    let hr = mk(device, bytes.as_ptr() as *const c_void, bytes.len(), std::ptr::null_mut(), &mut shader);
    if hr < 0 || shader.is_null() {
        return Err(format!("CreateComputeShader({name}) failed: hr={hr:#x}"));
    }
    Ok(shader)
}

unsafe fn compile_cs5(device: *mut c_void, compile: CompileFn, name: &str, source: &str) -> Result<*mut c_void, String> {
    let filename = CString::new(name).unwrap();
    let entry = CString::new("main").unwrap();
    let target = CString::new(SHADER_TARGET).unwrap();
    let mut code = std::ptr::null_mut();
    let mut errors = std::ptr::null_mut();
    let mut include = Include { vtbl: &INCLUDE_VTBL };
    let flags = D3DCOMPILE_OPTIMIZATION_LEVEL3
        | D3DCOMPILE_IEEE_STRICTNESS
        | D3DCOMPILE_ENABLE_STRICTNESS
        | D3DCOMPILE_WARNINGS_ARE_ERRORS;
    let hr = compile(
        source.as_ptr(),
        source.len(),
        filename.as_ptr(),
        std::ptr::null(),
        &mut include as *mut Include as *mut c_void,
        entry.as_ptr(),
        target.as_ptr(),
        flags,
        0,
        &mut code,
        &mut errors,
    );
    if hr < 0 || code.is_null() {
        let msg = blob_string(errors);
        release(errors);
        release(code);
        return Err(format!("D3DCompile {name} as {SHADER_TARGET} failed: hr={hr:#x} {msg}"));
    }
    type GetPtr = unsafe extern "system" fn(*mut c_void) -> *const c_void;
    type GetLen = unsafe extern "system" fn(*mut c_void) -> usize;
    let get_ptr: GetPtr = vslot(code, 3);
    let get_len: GetLen = vslot(code, 4);
    let bytes = get_ptr(code);
    let len = get_len(code);
    type CreateCs = unsafe extern "system" fn(*mut c_void, *const c_void, usize, *mut c_void, *mut *mut c_void) -> i32;
    let mk: CreateCs = vslot(device, 18);
    let mut shader = std::ptr::null_mut();
    let hr = mk(device, bytes, len, std::ptr::null_mut(), &mut shader);
    release(errors);
    release(code);
    if hr < 0 || shader.is_null() {
        return Err(format!("CreateComputeShader({name}) failed: hr={hr:#x}"));
    }
    Ok(shader)
}

unsafe fn adapter_identity(device: *mut c_void) -> Result<(u32, u32, String), String> {
    type Qi = unsafe extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> i32;
    let qi: Qi = vslot(device, 0);
    let mut dxgi = std::ptr::null_mut();
    let hr = qi(device, &IID_IDXGI_DEVICE, &mut dxgi);
    if hr < 0 || dxgi.is_null() {
        return Err(format!("IDXGIDevice query failed: hr={hr:#x}"));
    }
    type GetAdapter = unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> i32;
    let get_adapter: GetAdapter = vslot(dxgi, 7);
    let mut adapter = std::ptr::null_mut();
    let hr = get_adapter(dxgi, &mut adapter);
    release(dxgi);
    if hr < 0 || adapter.is_null() {
        return Err(format!("GetAdapter failed: hr={hr:#x}"));
    }
    #[repr(C)]
    struct Desc {
        description: [u16; 128],
        vendor: u32,
        device: u32,
        subsys: u32,
        revision: u32,
        dedicated_video: usize,
        dedicated_system: usize,
        shared_system: usize,
        luid_low: u32,
        luid_high: i32,
    }
    type GetDesc = unsafe extern "system" fn(*mut c_void, *mut Desc) -> i32;
    let get_desc: GetDesc = vslot(adapter, 8);
    let mut desc = std::mem::zeroed::<Desc>();
    let hr = get_desc(adapter, &mut desc);
    release(adapter);
    if hr < 0 {
        return Err(format!("GetDesc failed: hr={hr:#x}"));
    }
    let name = String::from_utf16_lossy(&desc.description)
        .trim_end_matches('\0')
        .to_string();
    Ok((desc.vendor, desc.device, name))
}

unsafe fn blob_string(blob: *mut c_void) -> String {
    if blob.is_null() {
        return String::new();
    }
    type GetPtr = unsafe extern "system" fn(*mut c_void) -> *const u8;
    type GetLen = unsafe extern "system" fn(*mut c_void) -> usize;
    let get_ptr: GetPtr = vslot(blob, 3);
    let get_len: GetLen = vslot(blob, 4);
    let ptr = get_ptr(blob);
    let len = get_len(blob);
    if ptr.is_null() || len == 0 {
        return String::new();
    }
    let bytes = std::slice::from_raw_parts(ptr, len);
    String::from_utf8_lossy(bytes).into_owned()
}

unsafe fn release(obj: *mut c_void) {
    if obj.is_null() {
        return;
    }
    type Release = unsafe extern "system" fn(*mut c_void) -> u32;
    let release: Release = vslot(obj, 2);
    release(obj);
}

unsafe fn vslot<T>(obj: *mut c_void, slot: usize) -> T {
    let vtbl = *(obj as *const *const usize);
    let fp = *vtbl.add(slot);
    std::mem::transmute_copy(&fp)
}

type CreateDevice = unsafe extern "system" fn(
    *mut c_void,
    u32,
    *mut c_void,
    u32,
    *const u32,
    u32,
    u32,
    *mut *mut c_void,
    *mut u32,
    *mut *mut c_void,
) -> i32;

type CompileFn = unsafe extern "system" fn(
    *const u8,
    usize,
    *const c_char,
    *const c_void,
    *mut c_void,
    *const c_char,
    *const c_char,
    u32,
    u32,
    *mut *mut c_void,
    *mut *mut c_void,
) -> i32;

#[repr(C)]
struct IncludeVtbl {
    open: unsafe extern "system" fn(*mut c_void, u32, *const c_char, *const c_void, *mut *const c_void, *mut u32) -> i32,
    close: unsafe extern "system" fn(*mut c_void, *const c_void) -> i32,
}

#[repr(C)]
struct Include {
    vtbl: *const IncludeVtbl,
}

unsafe extern "system" fn include_open(
    _this: *mut c_void,
    _kind: u32,
    name: *const c_char,
    _parent: *const c_void,
    out_data: *mut *const c_void,
    out_len: *mut u32,
) -> i32 {
    if name.is_null() {
        return -2147467259;
    }
    let text = CStr::from_ptr(name).to_string_lossy();
    let data: &'static str = if text.contains("common.hlsli") {
        COMMON_HLSLI
    } else if text.contains("probe_fs_q2") {
        PROBE_FS_Q2
    } else {
        return -2147467259;
    };
    *out_data = data.as_ptr() as *const c_void;
    *out_len = data.len() as u32;
    0
}

unsafe extern "system" fn include_close(_this: *mut c_void, _data: *const c_void) -> i32 {
    0
}

#[cfg(test)]
mod copy_tests {
    use super::copy_plane_rows;

    #[test]
    fn pitched_rows_match_a_scalar_copy() {
        let row = 640usize;
        let pitch = 768usize;
        let height = 36usize;
        let src: Vec<u8> = (0..row * height).map(|i| (i * 17 + 3) as u8).collect();
        let mut got = vec![0xA5u8; pitch * height];
        let mut expect = vec![0xA5u8; pitch * height];
        unsafe { copy_plane_rows(src.as_ptr(), got.as_mut_ptr(), row, pitch, height) };
        for y in 0..height {
            expect[y * pitch..y * pitch + row].copy_from_slice(&src[y * row..y * row + row]);
        }
        assert_eq!(got, expect);
    }

    #[test]
    fn streaming_store_matches_the_source() {
        let n = 1280usize * 24;
        let src: Vec<u8> = (0..n).map(|i| (i * 13 + 9) as u8).collect();
        let mut dst = vec![0u8; n + 64];
        let offset = 3usize;
        unsafe { super::avx_plane(src.as_ptr(), dst.as_mut_ptr().add(offset), n) };
        assert_eq!(&dst[offset..offset + n], &src[..]);
    }

    #[test]
    fn streaming_nt_matches_the_source_at_both_alignments() {
        let n = 1280usize * 8;
        let src: Vec<u8> = (0..n).map(|i| (i * 19 + 5) as u8).collect();
        for offset in [0usize, 3, 32] {
            let mut dst = vec![0x5Au8; n + 64];
            unsafe { super::copy_nt(src.as_ptr(), dst.as_mut_ptr().add(offset), n) };
            assert_eq!(&dst[offset..offset + n], &src[..], "offset {offset}");
            if offset > 0 {
                assert!(dst[..offset].iter().all(|&b| b == 0x5A));
            }
        }
    }
}

const INCLUDE_VTBL: IncludeVtbl = IncludeVtbl {
    open: include_open,
    close: include_close,
};

#[cfg(unix)]
unsafe fn open_library(name: &str) -> Result<*mut c_void, String> {
    #[link(name = "dl")]
    extern "C" {
        fn dlopen(filename: *const c_char, flags: i32) -> *mut c_void;
        fn dlerror() -> *mut c_char;
    }
    let c = CString::new(name).unwrap();
    let handle = dlopen(c.as_ptr(), 2);
    if handle.is_null() {
        let err = dlerror();
        let detail = if err.is_null() {
            format!("dlopen {name} returned null")
        } else {
            CStr::from_ptr(err).to_string_lossy().into_owned()
        };
        Err(detail)
    } else {
        Ok(handle)
    }
}

#[cfg(unix)]
unsafe fn lib_sym<T>(handle: *mut c_void, name: &str) -> Result<T, String> {
    #[link(name = "dl")]
    extern "C" {
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }
    let c = CString::new(name).unwrap();
    let p = dlsym(handle, c.as_ptr());
    if p.is_null() {
        Err(format!("missing {name}"))
    } else {
        Ok(std::mem::transmute_copy(&p))
    }
}

#[cfg(unix)]
unsafe fn free_library(handle: *mut c_void) {
    if handle.is_null() {
        return;
    }
    #[link(name = "dl")]
    extern "C" {
        fn dlclose(handle: *mut c_void) -> i32;
    }
    dlclose(handle);
}

#[cfg(target_os = "windows")]
unsafe fn open_library(name: &str) -> Result<*mut c_void, String> {
    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryA(name: *const c_char) -> *mut c_void;
        fn GetLastError() -> u32;
    }
    let c = CString::new(name).unwrap();
    let handle = LoadLibraryA(c.as_ptr());
    if handle.is_null() {
        Err(format!("LoadLibrary {name} failed: GetLastError={}", GetLastError()))
    } else {
        Ok(handle)
    }
}

#[cfg(target_os = "windows")]
unsafe fn lib_sym<T>(handle: *mut c_void, name: &str) -> Result<T, String> {
    #[link(name = "kernel32")]
    extern "system" {
        fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
    }
    let c = CString::new(name).unwrap();
    let p = GetProcAddress(handle, c.as_ptr());
    if p.is_null() {
        Err(format!("missing {name}"))
    } else {
        Ok(std::mem::transmute_copy(&p))
    }
}

#[cfg(target_os = "windows")]
unsafe fn free_library(handle: *mut c_void) {
    if handle.is_null() {
        return;
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn FreeLibrary(module: *mut c_void) -> i32;
    }
    FreeLibrary(handle);
}

#[cfg(not(any(unix, target_os = "windows")))]
unsafe fn open_library(name: &str) -> Result<*mut c_void, String> {
    let _ = name;
    Err("d3d11.dll is not loadable on this target".into())
}
#[cfg(not(any(unix, target_os = "windows")))]
unsafe fn lib_sym<T>(handle: *mut c_void, name: &str) -> Result<T, String> {
    let _ = (handle, name);
    Err("no loader".into())
}
#[cfg(not(any(unix, target_os = "windows")))]
unsafe fn free_library(handle: *mut c_void) {
    let _ = handle;
}

