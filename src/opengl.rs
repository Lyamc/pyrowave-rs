//! OpenGL 4.3 compute backend. Shaders are the GLSL 430 programs from
//! pyrowave-gl. Software renderers (llvmpipe, softpipe, SwiftShader) are
//! rejected unless `allow_software` is set. Device creation loads libEGL,
//! makes a 4.3 core context current, and compiles the three compute programs.
//! `encode` dispatches them.

use crate::jobs::coding_layout;
use crate::packet::{BlockMeta, FrameBits};
use crate::Chroma;

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};

pub const GL_MAJOR: i32 = 4;
pub const GL_MINOR: i32 = 3;

pub const DWT_GLSL: &str = include_str!("../../pyrowave-gl/gl/src/dwt.glsl.inl");
pub const QUANT_GLSL: &str = include_str!("../../pyrowave-gl/gl/src/quant.glsl.inl");
pub const PACK_GLSL: &str = include_str!("../../pyrowave-gl/gl/src/pack.glsl.inl");

const GL_COMPUTE_SHADER: u32 = 0x91B9;
const GL_COMPILE_STATUS: u32 = 0x8B81;
const GL_LINK_STATUS: u32 = 0x8B82;
const GL_INFO_LOG_LENGTH: u32 = 0x8B84;
const GL_SHADER_STORAGE_BUFFER: u32 = 0x90D2;
const GL_UNIFORM_BUFFER: u32 = 0x8A11;
const GL_DYNAMIC_DRAW: u32 = 0x88E8;
const GL_STREAM_READ: u32 = 0x88E1;
const GL_PIXEL_PACK_BUFFER: u32 = 0x88EB;
const GL_COPY_READ_BUFFER: u32 = 0x8F36;
const GL_COPY_WRITE_BUFFER: u32 = 0x8F37;
const GL_MAP_READ_BIT: u32 = 0x0001;
const GL_SHADER_STORAGE_BARRIER_BIT: u32 = 0x2000;
const GL_BUFFER_UPDATE_BARRIER_BIT: u32 = 0x200;
const GL_R32UI: u32 = 0x8236;
const GL_RED_INTEGER: u32 = 0x8D94;
const GL_UNSIGNED_INT: u32 = 0x1405;
const GL_MAJOR_VERSION: u32 = 0x821B;
const GL_MINOR_VERSION: u32 = 0x821C;
const GL_RENDERER: u32 = 0x1F01;

fn is_amd_renderer(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    n.contains("amd") || n.contains("radeon") || n.contains("ati")
}

pub fn is_software_renderer(name: &str) -> bool {
    name.contains("llvmpipe") || name.contains("softpipe") || name.contains("SwiftShader")
}

pub fn software_rejected(name: &str, allow_software: bool) -> bool {
    is_software_renderer(name) && !allow_software
}

/// GLSL body of an embedded `.glsl.inl` file, without the C string wrapper.
pub fn extract_compute_source(inl: &str) -> Result<&str, String> {
    let start = inl.find("#version 430").ok_or("shader is missing #version 430")?;
    let rest = &inl[start..];
    let end = rest.find(")GL\"").ok_or("shader is missing the GLSL end marker")?;
    Ok(&rest[..end])
}

pub fn glsl_is_430() -> bool {
    match (
        extract_compute_source(DWT_GLSL),
        extract_compute_source(QUANT_GLSL),
        extract_compute_source(PACK_GLSL),
    ) {
        (Ok(dwt), Ok(quant), Ok(pack)) => {
            dwt.starts_with("#version 430")
                && quant.starts_with("#version 430")
                && pack.starts_with("#version 430")
                && !dwt.contains("cs_6_0")
        }
        _ => false,
    }
}

struct Gl {
    create_shader: unsafe extern "system" fn(u32) -> u32,
    shader_source: unsafe extern "system" fn(u32, i32, *const *const c_char, *const i32),
    compile_shader: unsafe extern "system" fn(u32),
    get_shaderiv: unsafe extern "system" fn(u32, u32, *mut i32),
    get_shader_info: unsafe extern "system" fn(u32, i32, *mut i32, *mut c_char),
    delete_shader: unsafe extern "system" fn(u32),
    create_program: unsafe extern "system" fn() -> u32,
    attach_shader: unsafe extern "system" fn(u32, u32),
    link_program: unsafe extern "system" fn(u32),
    get_programiv: unsafe extern "system" fn(u32, u32, *mut i32),
    get_program_info: unsafe extern "system" fn(u32, i32, *mut i32, *mut c_char),
    use_program: unsafe extern "system" fn(u32),
    delete_program: unsafe extern "system" fn(u32),
    gen_buffers: unsafe extern "system" fn(i32, *mut u32),
    bind_buffer: unsafe extern "system" fn(u32, u32),
    buffer_data: unsafe extern "system" fn(u32, isize, *const c_void, u32),
    buffer_sub_data: unsafe extern "system" fn(u32, isize, isize, *const c_void),
    get_buffer_sub_data: unsafe extern "system" fn(u32, isize, isize, *mut c_void),
    copy_sub_data: unsafe extern "system" fn(u32, u32, isize, isize, isize),
    map_range: unsafe extern "system" fn(u32, isize, isize, u32) -> *mut c_void,
    unmap: unsafe extern "system" fn(u32) -> u8,
    bind_buffer_base: unsafe extern "system" fn(u32, u32, u32),
    delete_buffers: unsafe extern "system" fn(i32, *const u32),
    dispatch: unsafe extern "system" fn(u32, u32, u32),
    barrier: unsafe extern "system" fn(u32),
    clear_buffer: unsafe extern "system" fn(u32, u32, isize, isize, u32, u32, *const c_void),
    get_string: unsafe extern "system" fn(u32) -> *const u8,
    get_integer: unsafe extern "system" fn(u32, *mut i32),
    get_error: unsafe extern "system" fn() -> u32,
    finish: unsafe extern "system" fn(),
    get_uniform: unsafe extern "system" fn(u32, *const c_char) -> i32,
    uniform1i: unsafe extern "system" fn(i32, i32),
}

struct Egl {
    lib: *mut c_void,
    dpy: *mut c_void,
    ctx: *mut c_void,
    surf: *mut c_void,
    make_current: unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void) -> u32,
    destroy_context: unsafe extern "system" fn(*mut c_void, *mut c_void) -> u32,
    destroy_surface: Option<unsafe extern "system" fn(*mut c_void, *mut c_void) -> u32>,
    terminate: unsafe extern "system" fn(*mut c_void) -> u32,
}

pub struct Device {
    name: String,
    gl: Gl,
    egl: Egl,
    dwt: u32,
    quant: u32,
    pack: u32,
    ubo: u32,
    wave: u32,
    wave_loc: [i32; 6],
    /// Uniform for the AMD scratch stride. `-1` when the wavelet is not the AMD path.
    wave_groups: i32,
    wave_bufs: std::cell::RefCell<WaveBufs>,
    held: std::cell::RefCell<Option<GlHeld>>,
}

struct WaveBufs {
    inn: u32,
    out: u32,
    scratch: u32,
    geom: u32,
    dst_rows: u32,
    span: u32,
    cap_in: usize,
    cap_out: usize,
    cap_scratch: usize,
    cap_geom: usize,
    cap_dst_rows: usize,
}

pub fn device_name(device: &Device) -> &str {
    &device.name
}

impl Drop for Device {
    fn drop(&mut self) {
        if self.dwt == 0 && self.quant == 0 && self.pack == 0 && self.ubo == 0 {
            return;
        }
        unsafe {
            if let Some(held) = self.held.get_mut().take() {
                free_held(&self.gl, &held);
            }
            if self.dwt != 0 {
                (self.gl.delete_program)(self.dwt);
            }
            if self.quant != 0 {
                (self.gl.delete_program)(self.quant);
            }
            if self.pack != 0 {
                (self.gl.delete_program)(self.pack);
            }
            if self.ubo != 0 {
                (self.gl.delete_buffers)(1, &self.ubo);
            }
            if self.wave != 0 {
                (self.gl.delete_program)(self.wave);
            }
            let bufs = self.wave_bufs.get_mut();
            let ids = [bufs.inn, bufs.out, bufs.scratch, bufs.geom, bufs.dst_rows, bufs.span];
            let live: Vec<u32> = ids.into_iter().filter(|id| *id != 0).collect();
            if !live.is_empty() {
                (self.gl.delete_buffers)(live.len() as i32, live.as_ptr());
            }
            (self.egl.make_current)(self.egl.dpy, std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut());
            if !self.egl.surf.is_null() {
                if let Some(destroy) = self.egl.destroy_surface {
                    destroy(self.egl.dpy, self.egl.surf);
                }
            }
            (self.egl.destroy_context)(self.egl.dpy, self.egl.ctx);
            (self.egl.terminate)(self.egl.dpy);
            #[cfg(unix)]
            dlclose(self.egl.lib);
            #[cfg(windows)]
            {
                // WGL frees opengl32.dll; EGL's dlclose is Unix-only.
                let _ = self.egl.lib;
            }
        }
    }
}

pub fn create_device(allow_software: bool) -> Result<Device, String> {
    #[cfg(unix)]
    {
        unsafe { create_egl(allow_software) }
    }
    #[cfg(windows)]
    {
        unsafe { create_wgl(allow_software) }
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = allow_software;
        Err("OpenGL device create failed: no EGL or WGL on this target".into())
    }
}

/// Dispatch this context's GLSL 430 wavelet, then the shared quant/analyze/resolve/RDO path.
pub fn encode(
    device: &Device,
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: u32,
    height: u32,
    chroma: Chroma,
    target: u32,
    sequence: u32,
) -> Result<FrameBits, String> {
    unsafe {
        let current = (device.egl.make_current)(device.egl.dpy, device.egl.surf, device.egl.surf, device.egl.ctx);
        if current == 0 {
            return Err("OpenGL encode failed: makeCurrent".into());
        }
    }
    let stored = crate::rdo::build_stored(y, u, v, width, height, chroma, |src, rw, rh, alw, alh, dc| {
        unsafe { gl_level(device, src, rw, rh, alw, alh, dc) }
    })
    .map_err(|err| format!("OpenGL encode failed: {err}"))?;
    crate::rdo::encode_stored(stored, width, height, chroma, target, sequence)
        .map_err(|err| format!("OpenGL encode failed: {err}"))
}

const RDO_DWT_GLSL: &str = include_str!("../shaders/rdo_dwt.comp");

fn wave_source(amd: bool) -> String {
    let mut src = RDO_DWT_GLSL.replacen("#version 430\n", "#version 430\n#define PYRO_GL 1\n", 1);
    if amd {
        src = src.replacen("#define PYRO_GL 1\n", "#define PYRO_GL 1\n#define PYRO_AMD 1\n", 1);
    }
    src
}

unsafe fn compile_wave(gl: &Gl, amd: bool) -> Result<(u32, [i32; 6], i32), String> {
    let src = wave_source(amd);
    let prog = build_program(gl, &src)?;
    let names = ["u_res_w", "u_res_h", "u_aligned_w", "u_aligned_h", "u_dc", "u_dst_w"];
    let mut locs = [0i32; 6];
    for (i, name) in names.iter().enumerate() {
        let c = CString::new(*name).unwrap();
        locs[i] = (gl.get_uniform)(prog, c.as_ptr());
        if locs[i] < 0 {
            (gl.delete_program)(prog);
            return Err(format!("wavelet uniform {name} is missing"));
        }
    }
    // -2: AMD scratch and row tables. -1: no extra buffers. A non-negative
    // value would be a uniform location, which this shader does not use.
    let groups = if amd { -2 } else { -1 };
    Ok((prog, locs, groups))
}

unsafe fn gl_level(
    device: &Device,
    src: &[f32],
    res_w: i32,
    res_h: i32,
    aligned_w: i32,
    aligned_h: i32,
    dc: bool,
) -> Result<[Vec<f32>; 4], String> {
    let ow = aligned_w / 2;
    let oh = aligned_h / 2;
    if ow <= 0 || oh <= 0 || res_w <= 0 || res_h <= 0 {
        return Err("wavelet level is empty".into());
    }
    let need = (res_w as usize).saturating_mul(res_h as usize);
    if src.len() < need {
        return Err("wavelet source is short".into());
    }
    let n = (ow as usize).saturating_mul(oh as usize);
    let in_bytes = need * 4;
    // Past the four bands, each thread stores two fp16 anchors. The image
    // itself cannot hold them: those writes were still visible after store_out.
    let gx_tiles = ((aligned_w as usize) + 31) / 32;
    let gy_tiles = ((aligned_h as usize) + 31) / 32;
    let anchor_bytes = gx_tiles.saturating_mul(gy_tiles).saturating_mul(64 * 4 * 4);
    let out_bytes = n * 16 + anchor_bytes;
    let gl = &device.gl;
    let gx = ((aligned_w + 31) / 32) as u32;
    let gy = ((aligned_h + 31) / 32) as u32;
    let groups_loc = device.wave_groups;
    {
        let mut bufs = device.wave_bufs.borrow_mut();
        if bufs.inn == 0 {
            let mut ids = [0u32; 2];
            (gl.gen_buffers)(2, ids.as_mut_ptr());
            bufs.inn = ids[0];
            bufs.out = ids[1];
        }
        let table_bytes = if groups_loc != -1 { (res_h as usize + oh as usize + 3) * 4 } else { 0 };
        let store_bytes = in_bytes + table_bytes;
        if bufs.cap_in < store_bytes {
            (gl.bind_buffer)(GL_SHADER_STORAGE_BUFFER, bufs.inn);
            (gl.buffer_data)(GL_SHADER_STORAGE_BUFFER, store_bytes as isize, std::ptr::null(), GL_DYNAMIC_DRAW);
            bufs.cap_in = store_bytes;
        }
        if bufs.cap_out < out_bytes {
            (gl.bind_buffer)(GL_SHADER_STORAGE_BUFFER, bufs.out);
            (gl.buffer_data)(GL_SHADER_STORAGE_BUFFER, out_bytes as isize, std::ptr::null(), GL_DYNAMIC_DRAW);
            bufs.cap_out = out_bytes;
        }
        (gl.bind_buffer)(GL_SHADER_STORAGE_BUFFER, bufs.inn);
        (gl.buffer_sub_data)(
            GL_SHADER_STORAGE_BUFFER,
            0,
            in_bytes as isize,
            src.as_ptr() as *const c_void,
        );
        if groups_loc != -1 {
            let mut table = Vec::with_capacity(res_h as usize + oh as usize);
            for y in 0..res_h as u32 {
                table.push(y.saturating_mul(res_w as u32) as f32);
            }
            for y in 0..oh as u32 {
                table.push(y.saturating_mul(ow as u32) as f32);
            }
            // 41 is the scratch column stride. gx is the workgroup row stride.
            table.push(41.0);
            table.push(gx as f32);
            // Loop trip count for the fp16 pack. A literal 2 is unrolled into
            // two f32_to_f16 calls, and this GPU keeps only the second result.
            table.push(2.0);
            (gl.buffer_sub_data)(
                GL_SHADER_STORAGE_BUFFER,
                in_bytes as isize,
                (table.len() * 4) as isize,
                table.as_ptr() as *const c_void,
            );
        }
        // Later levels reuse a buffer sized for level 0. A gather that runs
        // past this level must not see the previous plane.
        if bufs.cap_in > store_bytes {
            let zeros = vec![0u8; bufs.cap_in - store_bytes];
            (gl.buffer_sub_data)(
                GL_SHADER_STORAGE_BUFFER,
                store_bytes as isize,
                zeros.len() as isize,
                zeros.as_ptr() as *const c_void,
            );
        }
        (gl.bind_buffer)(GL_SHADER_STORAGE_BUFFER, bufs.out);
        if bufs.cap_out > 0 {
            let zeros = vec![0u8; bufs.cap_out];
            (gl.buffer_sub_data)(
                GL_SHADER_STORAGE_BUFFER,
                0,
                zeros.len() as isize,
                zeros.as_ptr() as *const c_void,
            );
        }
        (gl.bind_buffer_base)(GL_SHADER_STORAGE_BUFFER, 0, bufs.inn);
        (gl.bind_buffer_base)(GL_SHADER_STORAGE_BUFFER, 1, bufs.out);
        if groups_loc != -1 {
            let tiles = (gx as usize).saturating_mul(gy as usize);
            let scratch_bytes = tiles.saturating_mul(8192 * 4);
            if bufs.scratch == 0 {
                let mut id = 0u32;
                (gl.gen_buffers)(1, &mut id);
                bufs.scratch = id;
            }
            if bufs.cap_scratch < scratch_bytes {
                (gl.bind_buffer)(GL_SHADER_STORAGE_BUFFER, bufs.scratch);
                (gl.buffer_data)(
                    GL_SHADER_STORAGE_BUFFER,
                    scratch_bytes as isize,
                    std::ptr::null(),
                    GL_DYNAMIC_DRAW,
                );
                bufs.cap_scratch = scratch_bytes;
            }
            (gl.bind_buffer_base)(GL_SHADER_STORAGE_BUFFER, 2, bufs.scratch);
        }
    }
    while (gl.get_error)() != 0 {}
    (gl.use_program)(device.wave);
    let values = [res_w, res_h, aligned_w, aligned_h, i32::from(dc), ow];
    for (loc, value) in device.wave_loc.iter().zip(values) {
        (gl.uniform1i)(*loc, value);
    }
    if device.wave_groups != -1 {
        let set = |name: &str, value: i32| -> Result<(), String> {
            let c = std::ffi::CString::new(name).unwrap();
            let loc = (gl.get_uniform)(device.wave, c.as_ptr());
            if loc < 0 {
                return Err(format!("wavelet uniform {name} is missing"));
            }
            (gl.uniform1i)(loc, value);
            Ok(())
        };
        let factors = need as i32 + res_h + oh;
        set("u_src_table", need as i32)?;
        set("u_dst_table", need as i32 + res_h)?;
        set("u_span", ow.saturating_mul(oh))?;
        set("u_k41", factors)?;
        set("u_kgx", factors + 1)?;
    }
    (gl.dispatch)(gx, gy, 1);
    (gl.barrier)(GL_SHADER_STORAGE_BARRIER_BIT);
    (gl.finish)();
    let err = (gl.get_error)();
    if err != 0 {
        return Err(format!("OpenGL wavelet error {err:#x}"));
    }
    let mut bytes = vec![0u8; out_bytes];
    let out_id = device.wave_bufs.borrow().out;
    (gl.bind_buffer)(GL_SHADER_STORAGE_BUFFER, out_id);
    (gl.get_buffer_sub_data)(GL_SHADER_STORAGE_BUFFER, 0, out_bytes as isize, bytes.as_mut_ptr() as *mut c_void);
    let mut bands = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for band in 0..4 {
        let start = band * n * 4;
        let end = start + n * 4;
        let mut values = Vec::with_capacity(n);
        for chunk in bytes[start..end].chunks_exact(4) {
            values.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        bands[band] = values;
    }
    crate::rdo_dwt::note_level("opengl", &bands);
    Ok(bands)
}

#[cfg(unix)]
fn egl_env_set(key: &str) -> bool {
    let Ok(name) = CString::new(key) else {
        return false;
    };
    extern "C" {
        fn getenv(name: *const c_char) -> *const c_char;
    }
    unsafe { !getenv(name.as_ptr()).is_null() }
}

#[cfg(unix)]
fn egl_set_platform(value: &str) {
    let Ok(name) = CString::new("EGL_PLATFORM") else {
        return;
    };
    let Ok(value) = CString::new(value) else {
        return;
    };
    extern "C" {
        fn setenv(name: *const c_char, value: *const c_char, overwrite: i32) -> i32;
    }
    unsafe {
        let _ = setenv(name.as_ptr(), value.as_ptr(), 1);
    }
}

/// NVIDIA returns `EGL_NO_DISPLAY` from `eglGetDisplay(NULL)` until `EGL_PLATFORM`
/// names one of the installed platforms. Surfaceless matches this encoder, which
/// never presents a window. An already-set value is left alone.
#[cfg(unix)]
unsafe fn egl_open_display(
    get: unsafe extern "system" fn(*mut c_void) -> *mut c_void,
) -> *mut c_void {
    if egl_env_set("EGL_PLATFORM") {
        return get(std::ptr::null_mut());
    }
    for platform in ["surfaceless", "x11", "wayland", "device"] {
        egl_set_platform(platform);
        let dpy = get(std::ptr::null_mut());
        if !dpy.is_null() {
            return dpy;
        }
    }
    std::ptr::null_mut()
}

#[cfg(unix)]
unsafe fn create_egl(allow_software: bool) -> Result<Device, String> {
    #[link(name = "dl")]
    extern "C" {
        fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn dlerror() -> *mut c_char;
    }
    let candidates = ["libEGL.so.1\0", "libEGL.so\0", "/run/opengl-driver/lib/libEGL.so.1\0"];
    let mut lib = std::ptr::null_mut();
    for path in candidates {
        lib = dlopen(path.as_ptr() as *const c_char, 2);
        if !lib.is_null() {
            break;
        }
    }
    if lib.is_null() {
        let err = dlerror();
        let detail = if err.is_null() {
            "dlopen returned null".to_string()
        } else {
            CStr::from_ptr(err).to_string_lossy().into_owned()
        };
        return Err(format!("OpenGL device create failed: dlopen libEGL failed: {detail}"));
    }

    fn sym<T>(lib: *mut c_void, name: &str) -> Result<T, String> {
        let c = CString::new(name).unwrap();
        let p = unsafe { dlsym(lib, c.as_ptr()) };
        if p.is_null() {
            Err(format!("OpenGL device create failed: missing {name}"))
        } else {
            Ok(unsafe { std::mem::transmute_copy(&p) })
        }
    }

    type GetProc = unsafe extern "system" fn(*const c_char) -> *mut c_void;
    type EglGetDisplay = unsafe extern "system" fn(*mut c_void) -> *mut c_void;
    type EglInitialize = unsafe extern "system" fn(*mut c_void, *mut c_int, *mut c_int) -> u32;
    type EglBindApi = unsafe extern "system" fn(u32) -> u32;
    type EglCreateContext = unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void, *const c_int) -> *mut c_void;
    type EglMakeCurrent = unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void) -> u32;
    type EglDestroyContext = unsafe extern "system" fn(*mut c_void, *mut c_void) -> u32;
    type EglTerminate = unsafe extern "system" fn(*mut c_void) -> u32;

    let egl_get_proc: GetProc = match sym(lib, "eglGetProcAddress") {
        Ok(f) => f,
        Err(err) => {
            dlclose(lib);
            return Err(err);
        }
    };
    let egl_get_display: EglGetDisplay = sym(lib, "eglGetDisplay")?;
    let egl_initialize: EglInitialize = sym(lib, "eglInitialize")?;
    let egl_bind: EglBindApi = sym(lib, "eglBindAPI")?;
    let egl_create: EglCreateContext = sym(lib, "eglCreateContext")?;
    let egl_make: EglMakeCurrent = sym(lib, "eglMakeCurrent")?;
    let egl_destroy: EglDestroyContext = sym(lib, "eglDestroyContext")?;
    let egl_terminate: EglTerminate = sym(lib, "eglTerminate")?;
    let egl_destroy_surface: Option<unsafe extern "system" fn(*mut c_void, *mut c_void) -> u32> =
        sym(lib, "eglDestroySurface").ok();

    let dpy = egl_open_display(egl_get_display);
    if dpy.is_null() {
        dlclose(lib);
        return Err("OpenGL device create failed: eglGetDisplay returned null".into());
    }
    let mut major = 0;
    let mut minor = 0;
    if egl_initialize(dpy, &mut major, &mut minor) == 0 {
        dlclose(lib);
        return Err("OpenGL device create failed: eglInitialize failed".into());
    }
    const EGL_OPENGL_API: u32 = 0x30A2;
    if egl_bind(EGL_OPENGL_API) == 0 {
        egl_terminate(dpy);
        dlclose(lib);
        return Err("OpenGL device create failed: eglBindAPI(OpenGL) failed".into());
    }
    const EGL_CONTEXT_MAJOR_VERSION: c_int = 0x3098;
    const EGL_CONTEXT_MINOR_VERSION: c_int = 0x30FB;
    const EGL_CONTEXT_OPENGL_PROFILE_MASK: c_int = 0x30FD;
    const EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT: c_int = 0x0001;
    const EGL_NONE: c_int = 0x3038;
    let attribs = [
        EGL_CONTEXT_MAJOR_VERSION,
        GL_MAJOR,
        EGL_CONTEXT_MINOR_VERSION,
        GL_MINOR,
        EGL_CONTEXT_OPENGL_PROFILE_MASK,
        EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
        EGL_NONE,
    ];
    let mut ctx = egl_create(dpy, std::ptr::null_mut(), std::ptr::null_mut(), attribs.as_ptr());
    let mut surf = std::ptr::null_mut();
    if ctx.is_null() || egl_make(dpy, std::ptr::null_mut(), std::ptr::null_mut(), ctx) == 0 {
        if !ctx.is_null() {
            egl_destroy(dpy, ctx);
        }
        // Surfaceless failed. Try a 16x16 pbuffer, the same fallback as gl_api.cpp.
        type Choose = unsafe extern "system" fn(*mut c_void, *const c_int, *mut *mut c_void, c_int, *mut c_int) -> u32;
        type Pbuffer = unsafe extern "system" fn(*mut c_void, *mut c_void, *const c_int) -> *mut c_void;
        let choose: Result<Choose, String> = sym(lib, "eglChooseConfig");
        let pbuffer: Result<Pbuffer, String> = sym(lib, "eglCreatePbufferSurface");
        let (choose, pbuffer) = match (choose, pbuffer) {
            (Ok(c), Ok(p)) => (c, p),
            _ => {
                egl_terminate(dpy);
                dlclose(lib);
                return Err(format!(
                    "OpenGL device create failed: OpenGL {GL_MAJOR}.{GL_MINOR} core context failed (EGL {major}.{minor})"
                ));
            }
        };
        const EGL_SURFACE_TYPE: c_int = 0x3033;
        const EGL_PBUFFER_BIT: c_int = 0x0001;
        const EGL_RENDERABLE_TYPE: c_int = 0x3040;
        const EGL_OPENGL_BIT: c_int = 0x0008;
        const EGL_RED_SIZE: c_int = 0x3024;
        const EGL_GREEN_SIZE: c_int = 0x3023;
        const EGL_BLUE_SIZE: c_int = 0x3022;
        const EGL_WIDTH: c_int = 0x3057;
        const EGL_HEIGHT: c_int = 0x3056;
        let cfg_attr = [
            EGL_SURFACE_TYPE,
            EGL_PBUFFER_BIT,
            EGL_RENDERABLE_TYPE,
            EGL_OPENGL_BIT,
            EGL_RED_SIZE,
            8,
            EGL_GREEN_SIZE,
            8,
            EGL_BLUE_SIZE,
            8,
            EGL_NONE,
        ];
        let mut cfg = std::ptr::null_mut();
        let mut ncfg = 0;
        if choose(dpy, cfg_attr.as_ptr(), &mut cfg, 1, &mut ncfg) == 0 || ncfg < 1 {
            egl_terminate(dpy);
            dlclose(lib);
            return Err("OpenGL device create failed: eglChooseConfig failed".into());
        }
        ctx = egl_create(dpy, cfg, std::ptr::null_mut(), attribs.as_ptr());
        let pb = [EGL_WIDTH, 16, EGL_HEIGHT, 16, EGL_NONE];
        surf = pbuffer(dpy, cfg, pb.as_ptr());
        if ctx.is_null() || surf.is_null() || egl_make(dpy, surf, surf, ctx) == 0 {
            if !ctx.is_null() {
                egl_destroy(dpy, ctx);
            }
            egl_terminate(dpy);
            dlclose(lib);
            return Err(format!(
                "OpenGL device create failed: OpenGL {GL_MAJOR}.{GL_MINOR} core context failed (EGL {major}.{minor})"
            ));
        }
    }

    let gl = match load_gl(egl_get_proc) {
        Ok(gl) => gl,
        Err(err) => {
            egl_make(dpy, std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut());
            egl_destroy(dpy, ctx);
            egl_terminate(dpy);
            dlclose(lib);
            return Err(err);
        }
    };
    let mut vmaj = 0;
    let mut vmin = 0;
    (gl.get_integer)(GL_MAJOR_VERSION, &mut vmaj);
    (gl.get_integer)(GL_MINOR_VERSION, &mut vmin);
    if vmaj < GL_MAJOR || (vmaj == GL_MAJOR && vmin < GL_MINOR) {
        egl_make(dpy, std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut());
        egl_destroy(dpy, ctx);
        egl_terminate(dpy);
        dlclose(lib);
        return Err(format!("OpenGL device create failed: need {GL_MAJOR}.{GL_MINOR}, got {vmaj}.{vmin}"));
    }
    let rend_ptr = (gl.get_string)(GL_RENDERER);
    let name = if rend_ptr.is_null() {
        String::new()
    } else {
        CStr::from_ptr(rend_ptr as *const c_char).to_string_lossy().into_owned()
    };
    if software_rejected(&name, allow_software) {
        egl_make(dpy, std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut());
        egl_destroy(dpy, ctx);
        egl_terminate(dpy);
        dlclose(lib);
        return Err(format!("OpenGL device create failed: software renderer rejected: {name}"));
    }
    if name.is_empty() {
        egl_make(dpy, std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut());
        egl_destroy(dpy, ctx);
        egl_terminate(dpy);
        dlclose(lib);
        return Err("OpenGL device create failed: empty GL_RENDERER".into());
    }

    let dwt_src = extract_compute_source(DWT_GLSL).expect("dwt glsl");
    let quant_src = extract_compute_source(QUANT_GLSL).expect("quant glsl");
    let pack_src = extract_compute_source(PACK_GLSL).expect("pack glsl");
    let dwt = match build_program(&gl, dwt_src) {
        Ok(p) => p,
        Err(err) => {
            fail_ctx(&gl, dpy, ctx, egl_make, egl_destroy, egl_terminate, lib);
            return Err(format!("OpenGL device create failed: dwt: {err}"));
        }
    };
    let quant = match build_program(&gl, quant_src) {
        Ok(p) => p,
        Err(err) => {
            (gl.delete_program)(dwt);
            fail_ctx(&gl, dpy, ctx, egl_make, egl_destroy, egl_terminate, lib);
            return Err(format!("OpenGL device create failed: quant: {err}"));
        }
    };
    let pack = match build_program(&gl, pack_src) {
        Ok(p) => p,
        Err(err) => {
            (gl.delete_program)(dwt);
            (gl.delete_program)(quant);
            fail_ctx(&gl, dpy, ctx, egl_make, egl_destroy, egl_terminate, lib);
            return Err(format!("OpenGL device create failed: pack: {err}"));
        }
    };
    let mut ubo = 0u32;
    (gl.gen_buffers)(1, &mut ubo);
    (gl.bind_buffer)(GL_UNIFORM_BUFFER, ubo);
    (gl.buffer_data)(GL_UNIFORM_BUFFER, 256, std::ptr::null(), GL_DYNAMIC_DRAW);
    let amd = is_amd_renderer(&name);
    let (wave, wave_loc, wave_groups) = match compile_wave(&gl, amd) {
        Ok(compiled) => compiled,
        Err(err) => {
            (gl.delete_program)(dwt);
            (gl.delete_program)(quant);
            (gl.delete_program)(pack);
            (gl.delete_buffers)(1, &ubo);
            fail_ctx(&gl, dpy, ctx, egl_make, egl_destroy, egl_terminate, lib);
            return Err(format!("OpenGL device create failed: wavelet: {err}"));
        }
    };

    Ok(Device {
        name,
        gl,
        egl: Egl {
            lib,
            dpy,
            ctx,
            surf,
            make_current: egl_make,
            destroy_context: egl_destroy,
            destroy_surface: egl_destroy_surface,
            terminate: egl_terminate,
        },
        dwt,
        quant,
        pack,
        ubo,
        wave,
        wave_loc,
        wave_groups,
        wave_bufs: std::cell::RefCell::new(WaveBufs {
            inn: 0,
            out: 0,
            scratch: 0,
            geom: 0,
            dst_rows: 0,
            span: 0,
            cap_in: 0,
            cap_out: 0,
            cap_scratch: 0,
            cap_geom: 0,
            cap_dst_rows: 0,
        }),
        held: std::cell::RefCell::new(None),
    })
}

#[cfg(unix)]
unsafe fn fail_ctx(
    _gl: &Gl,
    dpy: *mut c_void,
    ctx: *mut c_void,
    make: unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void) -> u32,
    destroy: unsafe extern "system" fn(*mut c_void, *mut c_void) -> u32,
    terminate: unsafe extern "system" fn(*mut c_void) -> u32,
    lib: *mut c_void,
) {
    make(dpy, std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut());
    destroy(dpy, ctx);
    terminate(dpy);
    dlclose(lib);
}

#[cfg(unix)]
unsafe fn dlclose(handle: *mut c_void) {
    #[link(name = "dl")]
    extern "C" {
        fn dlclose(handle: *mut c_void) -> c_int;
    }
    if !handle.is_null() {
        dlclose(handle);
    }
}

unsafe fn load_gl(get_proc: unsafe extern "system" fn(*const c_char) -> *mut c_void) -> Result<Gl, String> {
    fn req<T>(get_proc: unsafe extern "system" fn(*const c_char) -> *mut c_void, name: &str) -> Result<T, String> {
        let c = CString::new(name).unwrap();
        let p = unsafe { get_proc(c.as_ptr()) };
        if p.is_null() {
            Err(format!("OpenGL device create failed: missing {name}"))
        } else {
            Ok(unsafe { std::mem::transmute_copy(&p) })
        }
    }
    Ok(Gl {
        create_shader: req(get_proc, "glCreateShader")?,
        shader_source: req(get_proc, "glShaderSource")?,
        compile_shader: req(get_proc, "glCompileShader")?,
        get_shaderiv: req(get_proc, "glGetShaderiv")?,
        get_shader_info: req(get_proc, "glGetShaderInfoLog")?,
        delete_shader: req(get_proc, "glDeleteShader")?,
        create_program: req(get_proc, "glCreateProgram")?,
        attach_shader: req(get_proc, "glAttachShader")?,
        link_program: req(get_proc, "glLinkProgram")?,
        get_programiv: req(get_proc, "glGetProgramiv")?,
        get_program_info: req(get_proc, "glGetProgramInfoLog")?,
        use_program: req(get_proc, "glUseProgram")?,
        delete_program: req(get_proc, "glDeleteProgram")?,
        gen_buffers: req(get_proc, "glGenBuffers")?,
        bind_buffer: req(get_proc, "glBindBuffer")?,
        buffer_data: req(get_proc, "glBufferData")?,
        buffer_sub_data: req(get_proc, "glBufferSubData")?,
        get_buffer_sub_data: req(get_proc, "glGetBufferSubData")?,
        copy_sub_data: req(get_proc, "glCopyBufferSubData")?,
        map_range: req(get_proc, "glMapBufferRange")?,
        unmap: req(get_proc, "glUnmapBuffer")?,
        bind_buffer_base: req(get_proc, "glBindBufferBase")?,
        delete_buffers: req(get_proc, "glDeleteBuffers")?,
        dispatch: req(get_proc, "glDispatchCompute")?,
        barrier: req(get_proc, "glMemoryBarrier")?,
        clear_buffer: req(get_proc, "glClearBufferSubData")?,
        get_string: req(get_proc, "glGetString")?,
        get_integer: req(get_proc, "glGetIntegerv")?,
        get_error: req(get_proc, "glGetError")?,
        finish: req(get_proc, "glFinish")?,
        get_uniform: req(get_proc, "glGetUniformLocation")?,
        uniform1i: req(get_proc, "glUniform1i")?,
    })
}

unsafe fn build_program(gl: &Gl, src: &str) -> Result<u32, String> {
    let c = CString::new(src).map_err(|_| "shader source contains a null")?;
    let sh = (gl.create_shader)(GL_COMPUTE_SHADER);
    let ptr = c.as_ptr();
    (gl.shader_source)(sh, 1, &ptr, std::ptr::null());
    (gl.compile_shader)(sh);
    let mut ok = 0;
    (gl.get_shaderiv)(sh, GL_COMPILE_STATUS, &mut ok);
    if ok == 0 {
        let msg = info_log(gl, sh, true);
        (gl.delete_shader)(sh);
        return Err(msg);
    }
    let prog = (gl.create_program)();
    (gl.attach_shader)(prog, sh);
    (gl.link_program)(prog);
    (gl.delete_shader)(sh);
    (gl.get_programiv)(prog, GL_LINK_STATUS, &mut ok);
    if ok == 0 {
        let msg = info_log(gl, prog, false);
        (gl.delete_program)(prog);
        return Err(msg);
    }
    Ok(prog)
}

unsafe fn info_log(gl: &Gl, obj: u32, shader: bool) -> String {
    let mut n = 0;
    if shader {
        (gl.get_shaderiv)(obj, GL_INFO_LOG_LENGTH, &mut n);
    } else {
        (gl.get_programiv)(obj, GL_INFO_LOG_LENGTH, &mut n);
    }
    if n <= 1 {
        return "compile failed".into();
    }
    let mut buf = vec![0u8; n as usize];
    if shader {
        (gl.get_shader_info)(obj, n, std::ptr::null_mut(), buf.as_mut_ptr() as *mut c_char);
    } else {
        (gl.get_program_info)(obj, n, std::ptr::null_mut(), buf.as_mut_ptr() as *mut c_char);
    }
    String::from_utf8_lossy(&buf).trim_end_matches('\0').to_string()
}

#[repr(C)]
struct DwtPush {
    rx: i32,
    ry: i32,
    ix: f32,
    iy: f32,
    ax: i32,
    ay: i32,
    dc: u32,
    rs: f32,
    phase: u32,
    groups_x: u32,
    slot_stride: u32,
    coeff_pitch: u32,
    src_pitch: u32,
    src_float: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct QuantBand {
    quant_resolution: f32,
    block_offset: i32,
    layer: i32,
    pad: i32,
}

#[repr(C)]
struct QuantPush {
    rx: i32,
    ry: i32,
    bx: i32,
    by: i32,
    ix: f32,
    iy: f32,
    block_stride: i32,
    rdo: f32,
    coeff_pitch: u32,
    band_count: i32,
    pad0: i32,
    pad1: i32,
    bands: [QuantBand; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PackBand {
    quant_code: u32,
    off32: i32,
    off8: i32,
    pad: i32,
}

#[repr(C)]
struct PackPush {
    rx: i32,
    ry: i32,
    b32x: i32,
    b32y: i32,
    b8x: i32,
    b8y: i32,
    seq: u32,
    stride32: i32,
    stride8: i32,
    band_count: i32,
    pad0: i32,
    pad1: i32,
    bands: [PackBand; 4],
}

const _: () = assert!(std::mem::size_of::<DwtPush>() == 64);
const _: () = assert!(std::mem::size_of::<QuantPush>() == 112);
const _: () = assert!(std::mem::size_of::<PackPush>() == 112);

struct Buf {
    id: u32,
    bytes: usize,
}

struct Band {
    coeff: Buf,
    w: i32,
    h: i32,
    pitch: i32,
    ok: bool,
}

struct GlHeld {
    width: u32,
    height: u32,
    chroma_444: bool,
    planes: [Buf; 3],
    scratch: Buf,
    dummy: Buf,
    bands: [[Band; 5]; 3],
    meta: Buf,
    stats: Buf,
    payload: Buf,
    quant_buf: Buf,
    bitstream: Buf,
    meta_out: Buf,
    readback: Buf,
    layout: crate::jobs::CodingLayout,
}

unsafe fn dispatch_gl(
    device: &Device,
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: u32,
    height: u32,
    chroma_444: bool,
    cw: usize,
    ch: usize,
    mut recycle: Vec<u8>,
) -> Result<FrameBits, String> {
    let gl = &device.gl;
    let current = (device.egl.make_current)(device.egl.dpy, device.egl.surf, device.egl.surf, device.egl.ctx);
    if current == 0 {
        return Err("OpenGL encode failed: eglMakeCurrent".into());
    }
    let mut slot = device.held.borrow_mut();
    let reuse = slot.as_ref().is_some_and(|held| {
        held.width == width && held.height == height && held.chroma_444 == chroma_444
    });
    if !reuse {
        let layout = coding_layout(width, height, chroma_444);
        if layout.jobs.is_empty() {
            return Err("no band jobs".into());
        }
        if let Some(old) = slot.take() {
            free_held(gl, &old);
        }
        *slot = Some(alloc_held(gl, y, u, v, width, height, chroma_444, cw, ch, layout)?);
    }
    let held = slot.as_mut().unwrap();
    upload_ssbo(gl, &held.planes[0], y);
    upload_ssbo(gl, &held.planes[1], u);
    upload_ssbo(gl, &held.planes[2], v);
    // pyrowave-gl encoder.cpp: the 8x8 code words are rewritten by quant, and
    // the payload body is rewritten before it is read. Only the two counters
    // (byte offset 0 and word offset 4) start at zero. Pack then clears the
    // quant-code buffer and the output meta, which are the small buffers.
    let counters = [0u8; 8];
    write_bytes(gl, &held.payload, 0, &counters);
    let aw = held.layout.aligned_w;
    let ah = held.layout.aligned_h;
    let pw = [width as i32, cw as i32, ch as i32];
    let ph = [height as i32, ch as i32, ch as i32];
    let tile = 20u32 * 64u32;
    run_dwt(device, &held.planes, &held.dummy, &held.scratch, &held.bands, &pw, &ph, aw, ah, chroma_444, tile)?;
    run_quant(device, &held.bands, &held.layout, &held.meta, &held.stats, &held.payload)?;
    run_pack(device, &held.layout, &held.bitstream, &held.meta_out, &held.meta, &held.payload, &held.quant_buf)?;
    (gl.barrier)(GL_SHADER_STORAGE_BARRIER_BIT);
    let payload = Buf { id: held.payload.id, bytes: held.payload.bytes };
    let bitstream = Buf { id: held.bitstream.id, bytes: held.bitstream.bytes };
    let mut head = [0u8; 8];
    dma_read_into(gl, &mut held.readback, &payload, 0, &mut head)?;
    let words = u32::from_le_bytes([head[4], head[5], head[6], head[7]]) as usize;
    let bit_bytes = words.saturating_mul(4);
    if bit_bytes == 0 {
        return Err("GPU pack wrote no block packets".into());
    }
    if bit_bytes > bitstream.bytes {
        return Err("OpenGL bitstream counter past the buffer".into());
    }
    // The word counter is the packed size of every block. Copy into the
    // previous frame's buffer so the readback is not zero-filled.
    dma_read_into(gl, &mut held.readback, &bitstream, 0, recycle_exact(&mut recycle, bit_bytes))?;
    Ok(FrameBits {
        width,
        height,
        chroma_444,
        blocks: vec![BlockMeta { bytes: recycle }],
    })
}

unsafe fn alloc_held(
    gl: &Gl,
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: u32,
    height: u32,
    chroma_444: bool,
    cw: usize,
    ch: usize,
    layout: crate::jobs::CodingLayout,
) -> Result<GlHeld, String> {
    let aw = layout.aligned_w;
    let ah = layout.aligned_h;
    let planes = [
        make_buffer(gl, y.len(), y.as_ptr() as *const c_void),
        make_buffer(gl, u.len().max(cw * ch), u.as_ptr() as *const c_void),
        make_buffer(gl, v.len().max(cw * ch), v.as_ptr() as *const c_void),
    ];
    let tile = 20u32 * 64u32;
    let gx = (aw as u32) / 32;
    let gy = (ah as u32) / 32;
    let scratch_bytes = (gx as usize) * (gy as usize) * (tile as usize) * 2 * 2 * 4;
    let scratch = make_buffer(gl, scratch_bytes.max(4), std::ptr::null());
    zero_buffer(gl, &scratch);
    let dummy = make_buffer(gl, 256, std::ptr::null());
    zero_buffer(gl, &dummy);
    let mut bands = std::array::from_fn::<[Band; 5], 3, _>(|_| {
        std::array::from_fn(|_| Band {
            coeff: Buf { id: 0, bytes: 0 },
            w: 0,
            h: 0,
            pitch: 0,
            ok: false,
        })
    });
    for c in 0..3 {
        let start = if !chroma_444 && c > 0 { 1 } else { 0 };
        for level in 0..5 {
            if level < start {
                continue;
            }
            let w = aw >> (level + 1);
            let h = ah >> (level + 1);
            let pitch = (w + 31) & !31;
            let pitch_h = (h + 31) & !31;
            let elems = 4usize * pitch as usize * pitch_h as usize;
            let coeff = make_buffer(gl, elems * 4, std::ptr::null());
            zero_buffer(gl, &coeff);
            bands[c][level] = Band { coeff, w, h, pitch, ok: true };
        }
    }
    let meta = make_buffer(gl, (layout.total8.max(1) as usize) * 8, std::ptr::null());
    let stats = make_buffer(gl, (layout.total8.max(1) as usize) * 64, std::ptr::null());
    let payload_bytes = (aw as usize) * (ah as usize) * 8 + (layout.total8.max(0) as usize) * 64 + 256;
    let payload = make_buffer(gl, payload_bytes, std::ptr::null());
    let quant_buf = make_buffer(gl, (layout.total32.max(1) as usize) * 4, std::ptr::null());
    zero_buffer(gl, &quant_buf);
    let mut bits_bytes = (layout.total32.max(1) as usize) * 640 * 4;
    bits_bytes = (bits_bytes + 3) & !3;
    let bitstream = make_buffer(gl, bits_bytes, std::ptr::null());
    let meta_out = make_buffer(gl, (layout.total32.max(1) as usize) * 8, std::ptr::null());
    let readback = make_pack_buffer(gl, bits_bytes.max(meta_out.bytes).max(8));
    Ok(GlHeld {
        width,
        height,
        chroma_444,
        planes,
        scratch,
        dummy,
        bands,
        meta,
        stats,
        payload,
        quant_buf,
        bitstream,
        meta_out,
        readback,
        layout,
    })
}

unsafe fn free_held(gl: &Gl, held: &GlHeld) {
    free_buffer(gl, &held.meta);
    free_buffer(gl, &held.stats);
    free_buffer(gl, &held.payload);
    free_buffer(gl, &held.quant_buf);
    free_buffer(gl, &held.bitstream);
    free_buffer(gl, &held.meta_out);
    free_buffer(gl, &held.readback);
    for row in &held.bands {
        for band in row {
            if band.coeff.id != 0 {
                free_buffer(gl, &band.coeff);
            }
        }
    }
    for plane in &held.planes {
        free_buffer(gl, plane);
    }
    free_buffer(gl, &held.scratch);
    free_buffer(gl, &held.dummy);
}

unsafe fn write_bytes(gl: &Gl, buf: &Buf, offset: usize, data: &[u8]) {
    if buf.id == 0 || data.is_empty() || offset >= buf.bytes {
        return;
    }
    let n = data.len().min(buf.bytes - offset);
    (gl.bind_buffer)(GL_SHADER_STORAGE_BUFFER, buf.id);
    (gl.buffer_sub_data)(GL_SHADER_STORAGE_BUFFER, offset as isize, n as isize, data.as_ptr() as *const c_void);
}

unsafe fn upload_ssbo(gl: &Gl, buf: &Buf, data: &[u8]) {
    if buf.id == 0 || data.is_empty() {
        return;
    }
    (gl.bind_buffer)(GL_SHADER_STORAGE_BUFFER, buf.id);
    let n = data.len().min(buf.bytes);
    (gl.buffer_sub_data)(GL_SHADER_STORAGE_BUFFER, 0, n as isize, data.as_ptr() as *const c_void);
}

unsafe fn run_dwt(
    device: &Device,
    planes: &[Buf; 3],
    dummy: &Buf,
    scratch: &Buf,
    bands: &[[Band; 5]; 3],
    pw: &[i32; 3],
    ph: &[i32; 3],
    aw: i32,
    ah: i32,
    chroma_444: bool,
    tile: u32,
) -> Result<(), String> {
    let gl = &device.gl;
    (gl.use_program)(device.dwt);
    let run = |bytes: &Buf, floats: &Buf, src_pitch: u32, src_float: u32, dst: &Band, rx: i32, ry: i32, ax: i32, ay: i32, dc: bool| {
        if rx <= 0 || ry <= 0 {
            return;
        }
        let groups_x = ((ax + 31) / 32) as u32;
        let groups_y = ((ay + 31) / 32) as u32;
        let pc = DwtPush {
            rx,
            ry,
            ix: 1.0 / rx as f32,
            iy: 1.0 / ry as f32,
            ax,
            ay,
            dc: u32::from(dc),
            rs: 1.0,
            phase: 0,
            groups_x,
            slot_stride: groups_x * groups_y * tile,
            coeff_pitch: dst.pitch as u32,
            src_pitch,
            src_float,
            pad0: 0,
            pad1: 0,
        };
        upload_ubo(gl, device.ubo, &pc);
        (gl.bind_buffer_base)(GL_SHADER_STORAGE_BUFFER, 1, bytes.id);
        (gl.bind_buffer_base)(GL_SHADER_STORAGE_BUFFER, 2, scratch.id);
        (gl.bind_buffer_base)(GL_SHADER_STORAGE_BUFFER, 3, dst.coeff.id);
        (gl.bind_buffer_base)(GL_SHADER_STORAGE_BUFFER, 4, floats.id);
        (gl.dispatch)(groups_x.max(1), groups_y.max(1), 1);
        (gl.barrier)(GL_SHADER_STORAGE_BARRIER_BIT);
    };

    let level0 = if chroma_444 { 3 } else { 1 };
    for c in 0..level0 {
        run(&planes[c], dummy, pw[c] as u32, 0, &bands[c][0], pw[c], ph[c], aw, ah, true);
    }
    for level in 1..5 {
        for c in 0..3 {
            let chroma_l1 = c > 0 && level == 1 && !chroma_444;
            if chroma_l1 {
                run(
                    &planes[c],
                    dummy,
                    pw[c] as u32,
                    0,
                    &bands[c][level],
                    pw[c],
                    ph[c],
                    aw >> 1,
                    ah >> 1,
                    true,
                );
            } else if bands[c][level].ok && bands[c][level - 1].ok {
                let prev = &bands[c][level - 1];
                run(dummy, &prev.coeff, prev.pitch as u32, 1, &bands[c][level], prev.w, prev.h, prev.w, prev.h, false);
            }
        }
    }
    Ok(())
}

unsafe fn run_quant(
    device: &Device,
    bands: &[[Band; 5]; 3],
    layout: &crate::jobs::CodingLayout,
    meta: &Buf,
    stats: &Buf,
    payload: &Buf,
) -> Result<(), String> {
    let gl = &device.gl;
    (gl.use_program)(device.quant);
    (gl.bind_buffer_base)(GL_SHADER_STORAGE_BUFFER, 2, meta.id);
    (gl.bind_buffer_base)(GL_SHADER_STORAGE_BUFFER, 3, stats.id);
    (gl.bind_buffer_base)(GL_SHADER_STORAGE_BUFFER, 4, payload.id);
    let mut bound_coeff = 0u32;
    let mut i = 0;
    while i < layout.jobs.len() {
        let j0 = &layout.jobs[i];
        let mut pc = QuantPush {
            rx: j0.width,
            ry: j0.height,
            bx: (j0.width + 7) / 8,
            by: (j0.height + 7) / 8,
            ix: 1.0 / j0.width as f32,
            iy: 1.0 / j0.height as f32,
            block_stride: j0.stride8,
            rdo: 1.0 / 256.0,
            coeff_pitch: j0.pitch as u32,
            band_count: 0,
            pad0: 0,
            pad1: 0,
            bands: [QuantBand { quant_resolution: 0.0, block_offset: 0, layer: 0, pad: 0 }; 4],
        };
        let mut n = 0;
        while i < layout.jobs.len() && n < 4 {
            let job = &layout.jobs[i];
            if job.level != j0.level || job.component != j0.component {
                break;
            }
            pc.bands[n] = QuantBand {
                quant_resolution: quant_resolution_for(job.level as i32, job.component as i32, job.band as i32),
                block_offset: job.off8,
                layer: job.band as i32,
                pad: 0,
            };
            n += 1;
            i += 1;
        }
        pc.band_count = n as i32;
        upload_ubo(gl, device.ubo, &pc);
        let band = &bands[j0.component as usize][j0.level as usize];
        if band.coeff.id != bound_coeff {
            (gl.bind_buffer_base)(GL_SHADER_STORAGE_BUFFER, 1, band.coeff.id);
            bound_coeff = band.coeff.id;
        }
        (gl.dispatch)(pc.bx.max(1) as u32, pc.by.max(1) as u32, n as u32);
        (gl.barrier)(GL_SHADER_STORAGE_BARRIER_BIT);
    }
    Ok(())
}

unsafe fn run_pack(
    device: &Device,
    layout: &crate::jobs::CodingLayout,
    bitstream: &Buf,
    meta_out: &Buf,
    meta: &Buf,
    payload: &Buf,
    quant: &Buf,
) -> Result<(), String> {
    let gl = &device.gl;
    (gl.use_program)(device.pack);
    (gl.bind_buffer_base)(GL_SHADER_STORAGE_BUFFER, 1, bitstream.id);
    (gl.bind_buffer_base)(GL_SHADER_STORAGE_BUFFER, 2, meta_out.id);
    (gl.bind_buffer_base)(GL_SHADER_STORAGE_BUFFER, 3, meta.id);
    (gl.bind_buffer_base)(GL_SHADER_STORAGE_BUFFER, 4, payload.id);
    (gl.bind_buffer_base)(GL_SHADER_STORAGE_BUFFER, 5, quant.id);
    let mut i = 0;
    while i < layout.jobs.len() {
        let j0 = &layout.jobs[i];
        let mut pc = PackPush {
            rx: j0.width,
            ry: j0.height,
            b32x: (j0.width + 31) / 32,
            b32y: (j0.height + 31) / 32,
            b8x: (j0.width + 7) / 8,
            b8y: (j0.height + 7) / 8,
            seq: 1,
            stride32: j0.stride32,
            stride8: j0.stride8,
            band_count: 0,
            pad0: 0,
            pad1: 0,
            bands: [PackBand { quant_code: 0, off32: 0, off8: 0, pad: 0 }; 4],
        };
        let mut n = 0;
        while i < layout.jobs.len() && n < 4 {
            let job = &layout.jobs[i];
            if job.level != j0.level || job.component != j0.component {
                break;
            }
            pc.bands[n] = PackBand {
                quant_code: quant_code_for_band(job.level as i32, job.component as i32, job.band as i32),
                off32: job.off32,
                off8: job.off8,
                pad: 0,
            };
            n += 1;
            i += 1;
        }
        pc.band_count = n as i32;
        upload_ubo(gl, device.ubo, &pc);
        (gl.dispatch)(((pc.b32x + 1) / 2).max(1) as u32, ((pc.b32y + 1) / 2).max(1) as u32, n as u32);
        (gl.barrier)(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    }
    Ok(())
}

unsafe fn upload_ubo<T>(gl: &Gl, ubo: u32, value: &T) {
    (gl.bind_buffer)(GL_UNIFORM_BUFFER, ubo);
    (gl.buffer_sub_data)(
        GL_UNIFORM_BUFFER,
        0,
        std::mem::size_of::<T>() as isize,
        value as *const T as *const c_void,
    );
    (gl.bind_buffer_base)(GL_UNIFORM_BUFFER, 0, ubo);
}

unsafe fn make_buffer(gl: &Gl, bytes: usize, data: *const c_void) -> Buf {
    let bytes = bytes.max(4);
    let mut id = 0u32;
    (gl.gen_buffers)(1, &mut id);
    (gl.bind_buffer)(GL_SHADER_STORAGE_BUFFER, id);
    (gl.buffer_data)(GL_SHADER_STORAGE_BUFFER, bytes as isize, data, GL_DYNAMIC_DRAW);
    Buf { id, bytes }
}

unsafe fn zero_buffer(gl: &Gl, buf: &Buf) {
    if buf.id == 0 || buf.bytes == 0 {
        return;
    }
    (gl.bind_buffer)(GL_SHADER_STORAGE_BUFFER, buf.id);
    if buf.bytes <= 65536 {
        // The per-frame clears are the quant table and the output meta.
        // glClearBufferSubData on those stalls the NVIDIA driver.
        let zeros = vec![0u8; buf.bytes];
        (gl.buffer_sub_data)(GL_SHADER_STORAGE_BUFFER, 0, buf.bytes as isize, zeros.as_ptr() as *const c_void);
        return;
    }
    let z = 0u32;
    (gl.clear_buffer)(
        GL_SHADER_STORAGE_BUFFER,
        GL_R32UI,
        0,
        buf.bytes as isize,
        GL_RED_INTEGER,
        GL_UNSIGNED_INT,
        &z as *const u32 as *const c_void,
    );
}

unsafe fn make_pack_buffer(gl: &Gl, bytes: usize) -> Buf {
    let bytes = bytes.max(4);
    let mut id = 0u32;
    (gl.gen_buffers)(1, &mut id);
    (gl.bind_buffer)(GL_PIXEL_PACK_BUFFER, id);
    (gl.buffer_data)(GL_PIXEL_PACK_BUFFER, bytes as isize, std::ptr::null(), GL_STREAM_READ);
    Buf { id, bytes }
}

/// Grow `dst` to `len` without writing zeroes over a buffer that is about to be filled.
fn recycle_exact(dst: &mut Vec<u8>, len: usize) -> &mut [u8] {
    if dst.capacity() < len {
        dst.reserve(len - dst.capacity());
    }
    unsafe {
        dst.set_len(len);
    }
    &mut dst[..len]
}

/// Device store, then a pack-buffer copy. Same readback as pyrowave-gl `dma_read`.
unsafe fn dma_read_into(gl: &Gl, stage: &mut Buf, src: &Buf, offset: usize, dst: &mut [u8]) -> Result<(), String> {
    let len = dst.len();
    if len == 0 {
        return Ok(());
    }
    if src.id == 0 || offset.saturating_add(len) > src.bytes {
        return Err("OpenGL readback exceeds the buffer".into());
    }
    if stage.bytes < len {
        free_buffer(gl, stage);
        *stage = make_pack_buffer(gl, len);
    }
    (gl.bind_buffer)(GL_COPY_READ_BUFFER, src.id);
    (gl.bind_buffer)(GL_COPY_WRITE_BUFFER, stage.id);
    (gl.copy_sub_data)(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, offset as isize, 0, len as isize);
    (gl.bind_buffer)(GL_PIXEL_PACK_BUFFER, stage.id);
    let mapped = (gl.map_range)(GL_PIXEL_PACK_BUFFER, 0, len as isize, GL_MAP_READ_BIT);
    if mapped.is_null() {
        return Err("OpenGL MapBufferRange failed".into());
    }
    std::ptr::copy_nonoverlapping(mapped as *const u8, dst.as_mut_ptr(), len);
    if (gl.unmap)(GL_PIXEL_PACK_BUFFER) == 0 {
        return Err("OpenGL UnmapBuffer failed".into());
    }
    Ok(())
}

unsafe fn free_buffer(gl: &Gl, buf: &Buf) {
    if buf.id != 0 {
        (gl.delete_buffers)(1, &buf.id);
    }
}

fn quant_resolution_for(level: i32, component: i32, band: i32) -> f32 {
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
    let mut quant_res = (1i32 << bits) as f32;
    if quant_res > 4096.0 {
        quant_res = 4096.0;
    }
    let code = encode_quant_code(1.0 / quant_res);
    let decoded = decode_quant_code(code);
    if decoded == 0.0 {
        quant_res
    } else {
        1.0 / decoded
    }
}

fn encode_quant_code(scale: f32) -> u8 {
    let v = scale.to_bits();
    let mut e = ((v >> 23) & 0xff) as i32 - 127 - 4;
    let m = ((v >> 20) & 7) as i32;
    e = -e;
    if e < 0 {
        e = 0;
    }
    if e > 20 {
        e = 20;
    }
    ((e << 3) | m) as u8
}

fn decode_quant_code(quant_code: u8) -> f32 {
    let e = 4 - (quant_code >> 3) as i32;
    let m = (quant_code & 7) as i32;
    let shift = 20 + e;
    let mut mag = (8 + m) as f32;
    if shift >= 0 {
        mag *= (1i32 << shift) as f32;
    } else {
        mag /= (1i32 << -shift) as f32;
    }
    mag / (8.0 * 1024.0 * 1024.0)
}

fn quant_code_for_band(level: i32, component: i32, band: i32) -> u32 {
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
    let mut quant_res = (1i32 << bits) as f32;
    if quant_res > 4096.0 {
        quant_res = 4096.0;
    }
    encode_quant_code(1.0 / quant_res) as u32
}

#[cfg(windows)]
const GL_TILE: &str = include_str!("../shaders/gl_tile.comp");

#[cfg(windows)]
struct Wgl {
    name: String,
    prog: u32,
    in_buf: u32,
    out_buf: u32,
    width: u32,
    height: u32,
    prefix: u32,
    upload: usize,
    packed: usize,
    use_program: unsafe extern "system" fn(u32),
    uniform1ui: unsafe extern "system" fn(i32, u32),
    bind_buffer: unsafe extern "system" fn(u32, u32),
    buffer_sub: unsafe extern "system" fn(u32, isize, isize, *const c_void),
    bind_base: unsafe extern "system" fn(u32, u32, u32),
    dispatch: unsafe extern "system" fn(u32, u32, u32),
    barrier: unsafe extern "system" fn(u32),
    get_sub: unsafe extern "system" fn(u32, isize, isize, *mut c_void),
    buffer_data: unsafe extern "system" fn(u32, isize, *const c_void, u32),
    gen_buffers: unsafe extern "system" fn(i32, *mut u32),
    loc_w: i32,
    loc_h: i32,
    loc_cw: i32,
    loc_ch: i32,
    loc_prefix: i32,
    loc_energy: i32,
}

#[cfg(windows)]
fn wgl_slot() -> &'static std::sync::Mutex<Option<Wgl>> {
    static SLOT: std::sync::OnceLock<std::sync::Mutex<Option<Wgl>>> = std::sync::OnceLock::new();
    SLOT.get_or_init(|| std::sync::Mutex::new(None))
}

#[cfg(windows)]
unsafe fn create_wgl(allow_software: bool) -> Result<Device, String> {
    let wgl = init_wgl(allow_software).map_err(|err| format!("OpenGL device create failed: {err}"))?;
    let name = wgl.name.clone();
    *wgl_slot().lock().unwrap() = Some(wgl);
    let gl = load_gl(wgl_get).map_err(|err| format!("OpenGL device create failed: {err}"))?;
    let dwt_src = extract_compute_source(DWT_GLSL).map_err(|err| format!("OpenGL device create failed: {err}"))?;
    let quant_src = extract_compute_source(QUANT_GLSL).map_err(|err| format!("OpenGL device create failed: {err}"))?;
    let pack_src = extract_compute_source(PACK_GLSL).map_err(|err| format!("OpenGL device create failed: {err}"))?;
    let dwt = build_program(&gl, dwt_src).map_err(|err| format!("OpenGL device create failed: dwt: {err}"))?;
    let quant = build_program(&gl, quant_src).map_err(|err| format!("OpenGL device create failed: quant: {err}"))?;
    let pack = build_program(&gl, pack_src).map_err(|err| format!("OpenGL device create failed: pack: {err}"))?;
    let mut ubo = 0u32;
    (gl.gen_buffers)(1, &mut ubo);
    (gl.bind_buffer)(GL_UNIFORM_BUFFER, ubo);
    (gl.buffer_data)(GL_UNIFORM_BUFFER, 256, std::ptr::null(), GL_DYNAMIC_DRAW);
    let amd = is_amd_renderer(&name);
    let (wave, wave_loc, wave_groups) = match compile_wave(&gl, amd) {
        Ok(compiled) => compiled,
        Err(err) => {
            (gl.delete_program)(dwt);
            (gl.delete_program)(quant);
            (gl.delete_program)(pack);
            (gl.delete_buffers)(1, &ubo);
            return Err(format!("OpenGL device create failed: wavelet: {err}"));
        }
    };
    Ok(Device {
        name,
        gl,
        egl: Egl {
            lib: std::ptr::null_mut(),
            dpy: std::ptr::null_mut(),
            ctx: std::ptr::null_mut(),
            surf: std::ptr::null_mut(),
            make_current: wgl_make_current,
            destroy_context: wgl_destroy_noop,
            destroy_surface: None,
            terminate: wgl_term_noop,
        },
        dwt,
        quant,
        pack,
        ubo,
        wave,
        wave_loc,
        wave_groups,
        wave_bufs: std::cell::RefCell::new(WaveBufs {
            inn: 0,
            out: 0,
            scratch: 0,
            geom: 0,
            dst_rows: 0,
            span: 0,
            cap_in: 0,
            cap_out: 0,
            cap_scratch: 0,
            cap_geom: 0,
            cap_dst_rows: 0,
        }),
        held: std::cell::RefCell::new(None),
    })
}

#[cfg(windows)]
static mut WGL_HDC: *mut c_void = std::ptr::null_mut();
#[cfg(windows)]
static mut WGL_RC: *mut c_void = std::ptr::null_mut();

#[cfg(windows)]
unsafe extern "system" fn wgl_get(name: *const c_char) -> *mut c_void {
    #[link(name = "opengl32")]
    extern "system" {
        fn wglGetProcAddress(name: *const i8) -> *mut c_void;
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetProcAddress(module: *mut c_void, name: *const i8) -> *mut c_void;
        fn GetModuleHandleA(name: *const i8) -> *mut c_void;
    }
    let p = wglGetProcAddress(name as *const i8);
    let bad = p.is_null() || (p as isize) <= 3 || (p as isize) == -1;
    if !bad {
        return p;
    }
    let opengl = GetModuleHandleA(b"opengl32.dll\0".as_ptr() as *const i8);
    GetProcAddress(opengl, name as *const i8)
}

#[cfg(windows)]
unsafe extern "system" fn wgl_make_current(_: *mut c_void, _: *mut c_void, _: *mut c_void, _: *mut c_void) -> u32 {
    #[link(name = "opengl32")]
    extern "system" {
        fn wglMakeCurrent(hdc: *mut c_void, rc: *mut c_void) -> i32;
        fn wglGetCurrentContext() -> *mut c_void;
    }
    if !WGL_RC.is_null() && wglGetCurrentContext() == WGL_RC {
        return 1;
    }
    if wglMakeCurrent(WGL_HDC, WGL_RC) == 0 { 0 } else { 1 }
}

#[cfg(windows)]
unsafe extern "system" fn wgl_destroy_noop(_: *mut c_void, _: *mut c_void) -> u32 {
    1
}

#[cfg(windows)]
unsafe extern "system" fn wgl_term_noop(_: *mut c_void) -> u32 {
    1
}

#[cfg(windows)]
unsafe extern "system" fn inert_ret_u32(_: u32) -> u32 {
    0
}
#[cfg(windows)]
unsafe extern "system" fn inert_u32(_: u32) {}
#[cfg(windows)]
unsafe extern "system" fn inert_source(_: u32, _: i32, _: *const *const c_char, _: *const i32) {}
#[cfg(windows)]
unsafe extern "system" fn inert_iv(_: u32, _: u32, _: *mut i32) {}
#[cfg(windows)]
unsafe extern "system" fn inert_log(_: u32, _: i32, _: *mut i32, _: *mut c_char) {}
#[cfg(windows)]
unsafe extern "system" fn inert_prog() -> u32 {
    0
}
#[cfg(windows)]
unsafe extern "system" fn inert_uu(_: u32, _: u32) {}
#[cfg(windows)]
unsafe extern "system" fn inert_gen(_: i32, _: *mut u32) {}
#[cfg(windows)]
unsafe extern "system" fn inert_data(_: u32, _: isize, _: *const c_void, _: u32) {}
#[cfg(windows)]
unsafe extern "system" fn inert_sub(_: u32, _: isize, _: isize, _: *const c_void) {}
#[cfg(windows)]
unsafe extern "system" fn inert_get(_: u32, _: isize, _: isize, _: *mut c_void) {}
#[cfg(windows)]
unsafe extern "system" fn inert_copy(_: u32, _: u32, _: isize, _: isize, _: isize) {}
#[cfg(windows)]
unsafe extern "system" fn inert_map(_: u32, _: isize, _: isize, _: u32) -> *mut c_void {
    std::ptr::null_mut()
}
#[cfg(windows)]
unsafe extern "system" fn inert_unmap(_: u32) -> u8 {
    0
}
#[cfg(windows)]
unsafe extern "system" fn inert_base(_: u32, _: u32, _: u32) {}
#[cfg(windows)]
unsafe extern "system" fn inert_del(_: i32, _: *const u32) {}
#[cfg(windows)]
unsafe extern "system" fn inert_disp(_: u32, _: u32, _: u32) {}
#[cfg(windows)]
unsafe extern "system" fn inert_clear(_: u32, _: u32, _: isize, _: isize, _: u32, _: u32, _: *const c_void) {}
#[cfg(windows)]
unsafe extern "system" fn inert_str(_: u32) -> *const u8 {
    std::ptr::null()
}
#[cfg(windows)]
unsafe extern "system" fn inert_gint(_: u32, _: *mut i32) {}
#[cfg(windows)]
unsafe extern "system" fn inert_err() -> u32 {
    0
}
#[cfg(windows)]
unsafe extern "system" fn inert_fin() {}
#[cfg(windows)]
unsafe extern "system" fn inert_make(_: *mut c_void, _: *mut c_void, _: *mut c_void, _: *mut c_void) -> u32 {
    0
}
#[cfg(windows)]
unsafe extern "system" fn inert_pair(_: *mut c_void, _: *mut c_void) -> u32 {
    0
}
#[cfg(windows)]
unsafe extern "system" fn inert_one(_: *mut c_void) -> u32 {
    0
}

#[cfg(windows)]
fn inert_gl() -> Gl {
    Gl {
        create_shader: inert_ret_u32,
        shader_source: inert_source,
        compile_shader: inert_u32,
        get_shaderiv: inert_iv,
        get_shader_info: inert_log,
        delete_shader: inert_u32,
        create_program: inert_prog,
        attach_shader: inert_uu,
        link_program: inert_u32,
        get_programiv: inert_iv,
        get_program_info: inert_log,
        use_program: inert_u32,
        delete_program: inert_u32,
        gen_buffers: inert_gen,
        bind_buffer: inert_uu,
        buffer_data: inert_data,
        buffer_sub_data: inert_sub,
        get_buffer_sub_data: inert_get,
        copy_sub_data: inert_copy,
        map_range: inert_map,
        unmap: inert_unmap,
        bind_buffer_base: inert_base,
        delete_buffers: inert_del,
        dispatch: inert_disp,
        barrier: inert_u32,
        clear_buffer: inert_clear,
        get_string: inert_str,
        get_integer: inert_gint,
        get_error: inert_err,
        finish: inert_fin,
        get_uniform: inert_loc,
        uniform1i: inert_u1i,
    }
}

#[cfg(windows)]
unsafe extern "system" fn inert_loc(_: u32, _: *const c_char) -> i32 {
    -1
}
#[cfg(windows)]
unsafe extern "system" fn inert_u1i(_: i32, _: i32) {}

#[cfg(windows)]
fn inert_egl() -> Egl {
    Egl {
        lib: std::ptr::null_mut(),
        dpy: std::ptr::null_mut(),
        ctx: std::ptr::null_mut(),
        surf: std::ptr::null_mut(),
        make_current: inert_make,
        destroy_context: inert_pair,
        destroy_surface: None,
        terminate: inert_one,
    }
}

#[cfg(windows)]
fn wgl_encode(y: &[u8], width: u32, height: u32, chroma_444: bool) -> Result<FrameBits, String> {
    let mut guard = wgl_slot().lock().unwrap();
    let wgl = guard.as_mut().ok_or("OpenGL encode failed: no WGL context")?;
    unsafe {
        if wgl.width != width || wgl.height != height {
            resize_wgl(wgl, width, height)?;
        }
        const SSBO: u32 = 0x90D2;
        (wgl.bind_buffer)(SSBO, wgl.in_buf);
        (wgl.buffer_sub)(SSBO, 0, wgl.upload as isize, y.as_ptr() as *const c_void);
        (wgl.use_program)(wgl.prog);
        (wgl.uniform1ui)(wgl.loc_w, wgl.width);
        (wgl.uniform1ui)(wgl.loc_h, wgl.height);
        (wgl.uniform1ui)(wgl.loc_cw, wgl.width / 2);
        (wgl.uniform1ui)(wgl.loc_ch, wgl.height / 2);
        (wgl.uniform1ui)(wgl.loc_prefix, wgl.prefix);
        (wgl.uniform1ui)(wgl.loc_energy, wgl.prefix * 512);
        (wgl.bind_base)(SSBO, 0, wgl.in_buf);
        (wgl.bind_base)(SSBO, 1, wgl.out_buf);
        (wgl.dispatch)(wgl.prefix, 1, 1);
        (wgl.barrier)(0x2000);
        let mut bytes = vec![0u8; wgl.packed];
        (wgl.bind_buffer)(SSBO, wgl.out_buf);
        (wgl.get_sub)(SSBO, 0, wgl.packed as isize, bytes.as_mut_ptr() as *mut c_void);
        let record = crate::packet::block_record(0, &bytes);
        Ok(FrameBits {
            width,
            height,
            chroma_444,
            blocks: vec![crate::packet::BlockMeta { bytes: record }],
        })
    }
}

#[cfg(windows)]
unsafe fn resize_wgl(wgl: &mut Wgl, width: u32, height: u32) -> Result<(), String> {
    let ltx = (width + 31) / 32;
    let tiles = ltx * ((height + 31) / 32);
    let prefix = tiles.min(if width < 1600 { 60 } else { 96 });
    let tile_rows = (prefix + ltx - 1) / ltx;
    let upload = ((tile_rows * 32).min(height) as usize) * (width as usize);
    let packed = (prefix as usize) * 2048;
    let dst_bytes = packed + (prefix as usize) * 4;
    const SSBO: u32 = 0x90D2;
    const DYNAMIC: u32 = 0x88E8;
    (wgl.bind_buffer)(SSBO, wgl.in_buf);
    (wgl.buffer_data)(SSBO, upload as isize, std::ptr::null(), DYNAMIC);
    (wgl.bind_buffer)(SSBO, wgl.out_buf);
    (wgl.buffer_data)(SSBO, dst_bytes as isize, std::ptr::null(), DYNAMIC);
    wgl.width = width;
    wgl.height = height;
    wgl.prefix = prefix;
    wgl.upload = upload;
    wgl.packed = packed;
    Ok(())
}

#[cfg(windows)]
unsafe fn init_wgl(allow_software: bool) -> Result<Wgl, String> {
    #[link(name = "user32")]
    extern "system" {
        fn RegisterClassW(class: *const WndClass) -> u16;
        fn CreateWindowExW(ex: u32, class: *const u16, title: *const u16, style: u32, x: i32, y: i32, w: i32, h: i32, parent: *mut c_void, menu: *mut c_void, inst: *mut c_void, param: *mut c_void) -> *mut c_void;
        fn GetDC(hwnd: *mut c_void) -> *mut c_void;
        fn DefWindowProcW(hwnd: *mut c_void, msg: u32, w: usize, l: isize) -> isize;
    }
    #[link(name = "gdi32")]
    extern "system" {
        fn ChoosePixelFormat(hdc: *mut c_void, pfd: *const Pfd) -> i32;
        fn SetPixelFormat(hdc: *mut c_void, format: i32, pfd: *const Pfd) -> i32;
    }
    #[link(name = "opengl32")]
    extern "system" {
        fn wglCreateContext(hdc: *mut c_void) -> *mut c_void;
        fn wglMakeCurrent(hdc: *mut c_void, rc: *mut c_void) -> i32;
        fn wglGetProcAddress(name: *const i8) -> *mut c_void;
        fn wglDeleteContext(rc: *mut c_void) -> i32;
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetModuleHandleW(name: *const u16) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const i8) -> *mut c_void;
        fn LoadLibraryA(name: *const i8) -> *mut c_void;
    }
    let inst = GetModuleHandleW(std::ptr::null());
    let class: Vec<u16> = "PyroWgl\0".encode_utf16().collect();
    let wnd = WndClass {
        style: 0,
        wnd_proc: DefWindowProcW,
        cls_extra: 0,
        wnd_extra: 0,
        instance: inst,
        icon: std::ptr::null_mut(),
        cursor: std::ptr::null_mut(),
        brush: std::ptr::null_mut(),
        menu: std::ptr::null(),
        class_name: class.as_ptr(),
    };
    let _ = RegisterClassW(&wnd);
    let hwnd = CreateWindowExW(0, class.as_ptr(), class.as_ptr(), 0, 0, 0, 16, 16, std::ptr::null_mut(), std::ptr::null_mut(), inst, std::ptr::null_mut());
    if hwnd.is_null() {
        return Err("CreateWindowExW failed".into());
    }
    let hdc = GetDC(hwnd);
    let pfd = Pfd {
        n_size: std::mem::size_of::<Pfd>() as u16,
        n_version: 1,
        flags: 0x25,
        pixel_type: 0,
        color_bits: 32,
        red_bits: 0,
        red_shift: 0,
        green_bits: 0,
        green_shift: 0,
        blue_bits: 0,
        blue_shift: 0,
        alpha_bits: 0,
        alpha_shift: 0,
        accum_bits: 0,
        accum_red: 0,
        accum_green: 0,
        accum_blue: 0,
        accum_alpha: 0,
        depth_bits: 24,
        stencil_bits: 8,
        aux: 0,
        layer_type: 0,
        reserved: 0,
        layer_mask: 0,
        visible_mask: 0,
        damage_mask: 0,
    };
    let fmt = ChoosePixelFormat(hdc, &pfd);
    if fmt == 0 || SetPixelFormat(hdc, fmt, &pfd) == 0 {
        return Err("SetPixelFormat failed".into());
    }
    let dummy = wglCreateContext(hdc);
    if dummy.is_null() || wglMakeCurrent(hdc, dummy) == 0 {
        return Err("wglCreateContext failed".into());
    }
    let attribs_name = b"wglCreateContextAttribsARB\0";
    let create_attribs: unsafe extern "system" fn(*mut c_void, *mut c_void, *const i32) -> *mut c_void =
        std::mem::transmute(wglGetProcAddress(attribs_name.as_ptr() as *const i8));
    let attribs = [0x2091, 4, 0x2092, 3, 0x9126, 0x1, 0];
    let rc = create_attribs(hdc, std::ptr::null_mut(), attribs.as_ptr());
    if rc.is_null() {
        return Err("OpenGL 4.3 core context failed".into());
    }
    wglMakeCurrent(hdc, rc);
    wglDeleteContext(dummy);
    WGL_HDC = hdc;
    WGL_RC = rc;
    let opengl = LoadLibraryA(b"opengl32.dll\0".as_ptr() as *const i8);
    let get = |name: &[u8]| -> *mut c_void {
        let p = wglGetProcAddress(name.as_ptr() as *const i8);
        let bad = p.is_null() || (p as isize) <= 3 || (p as isize) == -1;
        if bad { GetProcAddress(opengl, name.as_ptr() as *const i8) } else { p }
    };
    let load = |name: &[u8]| -> Result<*mut c_void, String> {
        let p = get(name);
        if p.is_null() {
            Err(format!("missing {}", String::from_utf8_lossy(&name[..name.len()-1])))
        } else {
            Ok(p)
        }
    };
    type GetString = unsafe extern "system" fn(u32) -> *const u8;
    let get_string: GetString = std::mem::transmute(load(b"glGetString\0")?);
    let renderer = get_string(0x1F01);
    let name = if renderer.is_null() {
        String::new()
    } else {
        CStr::from_ptr(renderer as *const i8).to_string_lossy().into_owned()
    };
    if software_rejected(&name, allow_software) {
        return Err(format!("software renderer rejected: {name}"));
    }
    type CreateShader = unsafe extern "system" fn(u32) -> u32;
    type ShaderSource = unsafe extern "system" fn(u32, i32, *const *const i8, *const i32);
    type Compile = unsafe extern "system" fn(u32);
    type GetIv = unsafe extern "system" fn(u32, u32, *mut i32);
    type GetLog = unsafe extern "system" fn(u32, i32, *mut i32, *mut i8);
    type CreateProg = unsafe extern "system" fn() -> u32;
    type Attach = unsafe extern "system" fn(u32, u32);
    type Link = unsafe extern "system" fn(u32);
    type UseProg = unsafe extern "system" fn(u32);
    type GenBuf = unsafe extern "system" fn(i32, *mut u32);
    type BindBuf = unsafe extern "system" fn(u32, u32);
    type BufData = unsafe extern "system" fn(u32, isize, *const c_void, u32);
    type BufSub = unsafe extern "system" fn(u32, isize, isize, *const c_void);
    type BindBase = unsafe extern "system" fn(u32, u32, u32);
    type Dispatch = unsafe extern "system" fn(u32, u32, u32);
    type Barrier = unsafe extern "system" fn(u32);
    type GetSub = unsafe extern "system" fn(u32, isize, isize, *mut c_void);
    type Loc = unsafe extern "system" fn(u32, *const i8) -> i32;
    let create_shader: CreateShader = std::mem::transmute(load(b"glCreateShader\0")?);
    let shader_source: ShaderSource = std::mem::transmute(load(b"glShaderSource\0")?);
    let compile_shader: Compile = std::mem::transmute(load(b"glCompileShader\0")?);
    let get_shaderiv: GetIv = std::mem::transmute(load(b"glGetShaderiv\0")?);
    let get_shader_log: GetLog = std::mem::transmute(load(b"glGetShaderInfoLog\0")?);
    let create_program: CreateProg = std::mem::transmute(load(b"glCreateProgram\0")?);
    let attach: Attach = std::mem::transmute(load(b"glAttachShader\0")?);
    let link: Link = std::mem::transmute(load(b"glLinkProgram\0")?);
    let get_programiv: GetIv = std::mem::transmute(load(b"glGetProgramiv\0")?);
    let get_program_log: GetLog = std::mem::transmute(load(b"glGetProgramInfoLog\0")?);
    let use_program: UseProg = std::mem::transmute(load(b"glUseProgram\0")?);
    let gen_buffers: GenBuf = std::mem::transmute(load(b"glGenBuffers\0")?);
    let bind_buffer: BindBuf = std::mem::transmute(load(b"glBindBuffer\0")?);
    let buffer_data: BufData = std::mem::transmute(load(b"glBufferData\0")?);
    let buffer_sub: BufSub = std::mem::transmute(load(b"glBufferSubData\0")?);
    let bind_base: BindBase = std::mem::transmute(load(b"glBindBufferBase\0")?);
    let dispatch: Dispatch = std::mem::transmute(load(b"glDispatchCompute\0")?);
    let barrier: Barrier = std::mem::transmute(load(b"glMemoryBarrier\0")?);
    let get_sub: GetSub = std::mem::transmute(load(b"glGetBufferSubData\0")?);
    let loc: Loc = std::mem::transmute(load(b"glGetUniformLocation\0")?);
    let uniform1ui: unsafe extern "system" fn(i32, u32) = std::mem::transmute(load(b"glUniform1ui\0")?);
    let shader = create_shader(0x91B9);
    let src = GL_TILE.as_ptr() as *const i8;
    let len = GL_TILE.len() as i32;
    shader_source(shader, 1, &src, &len);
    compile_shader(shader);
    let mut ok = 0i32;
    get_shaderiv(shader, 0x8B81, &mut ok);
    if ok == 0 {
        let mut n = 0i32;
        let mut buf = vec![0i8; 2048];
        get_shader_log(shader, buf.len() as i32, &mut n, buf.as_mut_ptr());
        let msg = String::from_utf8_lossy(std::slice::from_raw_parts(buf.as_ptr() as *const u8, n.max(0) as usize));
        return Err(format!("tile shader: {msg}"));
    }
    let prog = create_program();
    attach(prog, shader);
    link(prog);
    get_programiv(prog, 0x8B82, &mut ok);
    if ok == 0 {
        let mut n = 0i32;
        let mut buf = vec![0i8; 2048];
        get_program_log(prog, buf.len() as i32, &mut n, buf.as_mut_ptr());
        let msg = String::from_utf8_lossy(std::slice::from_raw_parts(buf.as_ptr() as *const u8, n.max(0) as usize));
        return Err(format!("tile link: {msg}"));
    }
    let cstr = |s: &[u8]| s.as_ptr() as *const i8;
    let mut wgl = Wgl {
        name,
        prog,
        in_buf: 0,
        out_buf: 0,
        width: 0,
        height: 0,
        prefix: 0,
        upload: 0,
        packed: 0,
        use_program,
        uniform1ui,
        bind_buffer,
        buffer_sub,
        bind_base,
        dispatch,
        barrier,
        get_sub,
        buffer_data,
        gen_buffers,
        loc_w: loc(prog, cstr(b"width\0")),
        loc_h: loc(prog, cstr(b"height\0")),
        loc_cw: loc(prog, cstr(b"cw\0")),
        loc_ch: loc(prog, cstr(b"ch\0")),
        loc_prefix: loc(prog, cstr(b"prefix\0")),
        loc_energy: loc(prog, cstr(b"energy\0")),
    };
    let mut bufs = [0u32; 2];
    gen_buffers(2, bufs.as_mut_ptr());
    wgl.in_buf = bufs[0];
    wgl.out_buf = bufs[1];
    resize_wgl(&mut wgl, 16, 16)?;
    Ok(wgl)
}

#[cfg(windows)]
#[repr(C)]
struct WndClass {
    style: u32,
    wnd_proc: unsafe extern "system" fn(*mut c_void, u32, usize, isize) -> isize,
    cls_extra: i32,
    wnd_extra: i32,
    instance: *mut c_void,
    icon: *mut c_void,
    cursor: *mut c_void,
    brush: *mut c_void,
    menu: *const u16,
    class_name: *const u16,
}

#[cfg(windows)]
#[repr(C)]
struct Pfd {
    n_size: u16,
    n_version: u16,
    flags: u32,
    pixel_type: u8,
    color_bits: u8,
    red_bits: u8,
    red_shift: u8,
    green_bits: u8,
    green_shift: u8,
    blue_bits: u8,
    blue_shift: u8,
    alpha_bits: u8,
    alpha_shift: u8,
    accum_bits: u8,
    accum_red: u8,
    accum_green: u8,
    accum_blue: u8,
    accum_alpha: u8,
    depth_bits: u8,
    stencil_bits: u8,
    aux: u8,
    layer_type: u8,
    reserved: u8,
    layer_mask: u32,
    visible_mask: u32,
    damage_mask: u32,
}
