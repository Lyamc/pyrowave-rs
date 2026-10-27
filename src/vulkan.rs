//! Vulkan compute encode. Loads the real loader, rejects lavapipe unless
//! `allow_software` is set, and dispatches the embedded CDF 9/7 tile shader.
//! Every luma and chroma tile is lifted, quantized, and packed.

use crate::packet::{block_record, BlockMeta, FrameBits};
use crate::Chroma;
use std::ffi::CString;
use std::os::raw::{c_char, c_void};

const SPIRV: &[u8] = include_bytes!("rdo_dwt.spv");

const API_1_0: u32 = 1 << 22;
const QUEUE_COMPUTE: u32 = 0x2;
const STAGE_COMPUTE: u32 = 0x20;
const STAGE_HOST: u32 = 0x4000;
const ACCESS_SHADER_READ: u32 = 0x20;
const ACCESS_SHADER_WRITE: u32 = 0x40;
const ACCESS_HOST_READ: u32 = 0x2000;
const ACCESS_HOST_WRITE: u32 = 0x4000;
const BUFFER_STORAGE: u32 = 0x20;
const BUFFER_TRANSFER_SRC: u32 = 0x1;
const BUFFER_TRANSFER_DST: u32 = 0x2;
const MEM_DEVICE_LOCAL: u32 = 0x1;
const MEM_HOST_VISIBLE: u32 = 0x2;
const MEM_HOST_COHERENT: u32 = 0x4;
const MEM_HOST_CACHED: u32 = 0x8;
const STAGE_TRANSFER: u32 = 0x1000;
const ACCESS_TRANSFER_READ: u32 = 0x800;
const ACCESS_TRANSFER_WRITE: u32 = 0x1000;
const DESC_STORAGE: u32 = 7;
const BIND_COMPUTE: u32 = 1;
const WHOLE: u64 = u64::MAX;

type GetProc = unsafe extern "system" fn(u64, *const c_char) -> *const c_void;
type FnCreateInstance = unsafe extern "system" fn(*const InstanceInfo, *const c_void, *mut u64) -> i32;
type FnEnumPhys = unsafe extern "system" fn(u64, *mut u32, *mut u64) -> i32;
type FnProps = unsafe extern "system" fn(u64, *mut u8);
type FnQueueFamilies = unsafe extern "system" fn(u64, *mut u32, *mut QueueFamily);
type FnCreateDevice = unsafe extern "system" fn(u64, *const DeviceInfo, *const c_void, *mut u64) -> i32;
type FnGetQueue = unsafe extern "system" fn(u64, u32, u32, *mut u64);
type FnMemProps = unsafe extern "system" fn(u64, *mut MemProps);
type FnCreateBuffer = unsafe extern "system" fn(u64, *const BufferInfo, *const c_void, *mut u64) -> i32;
type FnDestroyBuffer = unsafe extern "system" fn(u64, u64, *const c_void);
type FnBufReqs = unsafe extern "system" fn(u64, u64, *mut MemReq);
type FnAlloc = unsafe extern "system" fn(u64, *const AllocInfo, *const c_void, *mut u64) -> i32;
type FnFreeMem = unsafe extern "system" fn(u64, u64, *const c_void);
type FnBind = unsafe extern "system" fn(u64, u64, u64, u64) -> i32;
type FnMap = unsafe extern "system" fn(u64, u64, u64, u64, u32, *mut *mut c_void) -> i32;
type FnCreateShader = unsafe extern "system" fn(u64, *const ShaderInfo, *const c_void, *mut u64) -> i32;
type FnSetLayout = unsafe extern "system" fn(u64, *const SetLayoutInfo, *const c_void, *mut u64) -> i32;
type FnPipeLayout = unsafe extern "system" fn(u64, *const PipeLayoutInfo, *const c_void, *mut u64) -> i32;
type FnComputePipe = unsafe extern "system" fn(u64, u64, u32, *const ComputeInfo, *const c_void, *mut u64) -> i32;
type FnPool = unsafe extern "system" fn(u64, *const PoolInfo, *const c_void, *mut u64) -> i32;
type FnAllocSets = unsafe extern "system" fn(u64, *const SetAlloc, *mut u64) -> i32;
type FnUpdateSets = unsafe extern "system" fn(u64, u32, *const WriteSet, u32, *const c_void);
type FnCmdPool = unsafe extern "system" fn(u64, *const CmdPoolInfo, *const c_void, *mut u64) -> i32;
type FnAllocCmd = unsafe extern "system" fn(u64, *const CmdAlloc, *mut u64) -> i32;
type FnFence = unsafe extern "system" fn(u64, *const FenceInfo, *const c_void, *mut u64) -> i32;
type FnBegin = unsafe extern "system" fn(u64, *const BeginInfo) -> i32;
type FnEnd = unsafe extern "system" fn(u64) -> i32;
type FnBindPipe = unsafe extern "system" fn(u64, u32, u64);
type FnBindSets = unsafe extern "system" fn(u64, u32, u64, u32, u32, *const u64, u32, *const u32);
type FnPush = unsafe extern "system" fn(u64, u64, u32, u32, u32, *const u8);
type FnDispatch = unsafe extern "system" fn(u64, u32, u32, u32);
type FnCopy = unsafe extern "system" fn(u64, u64, u64, u32, *const CopyRegion);
type FnBarrier = unsafe extern "system" fn(u64, u32, u32, u32, u32, *const c_void, u32, *const BufBarrier, u32, *const c_void);
type FnResetFence = unsafe extern "system" fn(u64, u32, *const u64) -> i32;
type FnSubmit = unsafe extern "system" fn(u64, u32, *const SubmitInfo, u64) -> i32;
type FnWait = unsafe extern "system" fn(u64, u32, *const u64, u32, u64) -> i32;
type FnResetPool = unsafe extern "system" fn(u64, u64, u32) -> i32;
type FnDestroyDevice = unsafe extern "system" fn(u64, *const c_void);
type FnDestroyInstance = unsafe extern "system" fn(u64, *const c_void);

#[repr(C)]
struct InstanceInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    app: *const AppInfo,
    layer_count: u32,
    layers: *const *const c_char,
    ext_count: u32,
    exts: *const *const c_char,
}

#[repr(C)]
struct AppInfo {
    s_type: u32,
    p_next: *const c_void,
    name: *const c_char,
    app_version: u32,
    engine: *const c_char,
    engine_version: u32,
    api: u32,
}

#[repr(C)]
struct QueueInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    family: u32,
    count: u32,
    priorities: *const f32,
}

#[repr(C)]
struct DeviceInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    queue_count: u32,
    queues: *const QueueInfo,
    layer_count: u32,
    layers: *const *const c_char,
    ext_count: u32,
    exts: *const *const c_char,
    features: *const c_void,
}

#[repr(C)]
#[derive(Clone)]
struct QueueFamily {
    flags: u32,
    count: u32,
    timestamp: u32,
    gx: u32,
    gy: u32,
    gz: u32,
}

#[repr(C)]
struct MemType {
    flags: u32,
    heap: u32,
}

#[repr(C)]
struct MemHeap {
    size: u64,
    flags: u32,
    _pad: u32,
}

#[repr(C)]
struct MemProps {
    type_count: u32,
    types: [MemType; 32],
    heap_count: u32,
    _pad: u32,
    heaps: [MemHeap; 16],
}

#[repr(C)]
struct BufferInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    size: u64,
    usage: u32,
    sharing: u32,
    family_count: u32,
    families: *const u32,
}

#[repr(C)]
struct MemReq {
    size: u64,
    alignment: u64,
    bits: u32,
    _pad: u32,
}

#[repr(C)]
struct AllocInfo {
    s_type: u32,
    p_next: *const c_void,
    size: u64,
    index: u32,
}

#[repr(C)]
struct Binding {
    binding: u32,
    desc_type: u32,
    count: u32,
    stages: u32,
    samplers: *const c_void,
}

#[repr(C)]
struct SetLayoutInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    count: u32,
    bindings: *const Binding,
}

#[repr(C)]
struct PushRange {
    stages: u32,
    offset: u32,
    size: u32,
}

#[repr(C)]
struct PipeLayoutInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    set_count: u32,
    sets: *const u64,
    push_count: u32,
    pushes: *const PushRange,
}

#[repr(C)]
struct ShaderInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    code_size: usize,
    code: *const u32,
}

#[repr(C)]
struct StageInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    stage: u32,
    module: u64,
    name: *const c_char,
    spec: *const c_void,
}

#[repr(C)]
struct ComputeInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    stage: StageInfo,
    layout: u64,
    base: u64,
    base_index: i32,
}

#[repr(C)]
struct PoolSize {
    desc_type: u32,
    count: u32,
}

#[repr(C)]
struct PoolInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    max_sets: u32,
    size_count: u32,
    sizes: *const PoolSize,
}

#[repr(C)]
struct SetAlloc {
    s_type: u32,
    p_next: *const c_void,
    pool: u64,
    count: u32,
    layouts: *const u64,
}

#[repr(C)]
struct BufDesc {
    buffer: u64,
    offset: u64,
    range: u64,
}

#[repr(C)]
struct WriteSet {
    s_type: u32,
    p_next: *const c_void,
    dst: u64,
    binding: u32,
    element: u32,
    count: u32,
    desc_type: u32,
    image: *const c_void,
    buffer: *const BufDesc,
    texel: *const c_void,
}

#[repr(C)]
struct CmdPoolInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    family: u32,
}

#[repr(C)]
struct CmdAlloc {
    s_type: u32,
    p_next: *const c_void,
    pool: u64,
    level: u32,
    count: u32,
}

#[repr(C)]
struct FenceInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
}

#[repr(C)]
struct BeginInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    inherit: *const c_void,
}

#[repr(C)]
struct BufBarrier {
    s_type: u32,
    p_next: *const c_void,
    src_access: u32,
    dst_access: u32,
    src_queue: u32,
    dst_queue: u32,
    buffer: u64,
    offset: u64,
    size: u64,
}

#[repr(C)]
struct CopyRegion {
    src: u64,
    dst: u64,
    size: u64,
}

#[repr(C)]
struct SubmitInfo {
    s_type: u32,
    p_next: *const c_void,
    wait_count: u32,
    waits: *const u64,
    wait_stages: *const u32,
    cmd_count: u32,
    cmds: *const u64,
    signal_count: u32,
    signals: *const u64,
}

#[repr(C)]
struct Push {
    width: u32,
    height: u32,
    cw: u32,
    ch: u32,
    prefix: u32,
    energy: u32,
}

#[repr(C)]
struct DwtPush {
    res_w: i32,
    res_h: i32,
    aligned_w: i32,
    aligned_h: i32,
    dc: i32,
    dst_w: i32,
}

struct Procs {
    destroy_instance: FnDestroyInstance,
    destroy_device: FnDestroyDevice,
    destroy_buffer: FnDestroyBuffer,
    free_mem: FnFreeMem,
    create_buffer: FnCreateBuffer,
    buf_reqs: FnBufReqs,
    alloc: FnAlloc,
    bind: FnBind,
    map: FnMap,
    alloc_sets: FnAllocSets,
    update_sets: FnUpdateSets,
    alloc_cmd: FnAllocCmd,
    begin: FnBegin,
    end: FnEnd,
    bind_pipe: FnBindPipe,
    bind_sets: FnBindSets,
    push: FnPush,
    dispatch: FnDispatch,
    copy: FnCopy,
    barrier: FnBarrier,
    reset_fence: FnResetFence,
    submit: FnSubmit,
    wait: FnWait,
    reset_pool: FnResetPool,
}

struct Gpu {
    #[allow(dead_code)]
    lib: *mut c_void,
    instance: u64,
    device: u64,
    queue: u64,
    layout: u64,
    pipeline: u64,
    pool: u64,
    set: u64,
    cmd_pool: u64,
    cmd: u64,
    fence: u64,
    family: u32,
    mem: MemProps,
    buf_in: u64,
    mem_in: u64,
    buf_out: u64,
    mem_out: u64,
    dev_in: u64,
    dev_in_mem: u64,
    dev_out: u64,
    dev_out_mem: u64,
    map_in: *mut u8,
    map_out: *mut u8,
    width: u32,
    height: u32,
    chroma_444: bool,
    prefix: u32,
    upload: usize,
    ready: bool,
    cap_in: u64,
    cap_out: u64,
    procs: Procs,
}

pub struct Device {
    name: String,
    gpu: std::cell::RefCell<Gpu>,
}

pub fn device_name(device: &Device) -> &str {
    &device.name
}

pub fn create_device(allow_software: bool) -> Result<Device, String> {
    create_device_inner(allow_software).map_err(|err| format!("Vulkan device create failed: {err}"))
}

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
    let stored = {
        let mut gpu = device.gpu.borrow_mut();
        crate::rdo::build_stored(y, u, v, width, height, chroma, |src, rw, rh, alw, alh, dc| {
            dispatch_level(&mut gpu, src, rw, rh, alw, alh, dc)
        })
    }
    .map_err(|err| format!("Vulkan encode failed: {err}"))?;
    crate::rdo::encode_stored(stored, width, height, chroma, target, sequence)
        .map_err(|err| format!("Vulkan encode failed: {err}"))
}

fn dispatch_level(
    gpu: &mut Gpu,
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
    let in_bytes = (need * 4) as u64;
    let out_bytes = (n * 4 * 4) as u64;
    unsafe {
        ensure_dwt_buffers(gpu, in_bytes, out_bytes)?;
        std::ptr::copy_nonoverlapping(src.as_ptr() as *const u8, gpu.map_in, in_bytes as usize);
        std::ptr::write_bytes(gpu.map_out, 0, out_bytes as usize);
        let push = DwtPush {
            res_w,
            res_h,
            aligned_w,
            aligned_h,
            dc: i32::from(dc),
            dst_w: ow,
        };
        let gx = ((aligned_w + 31) / 32) as u32;
        let gy = ((aligned_h + 31) / 32) as u32;
        submit_level(gpu, &push, gx, gy)?;
        let raw = std::slice::from_raw_parts(gpu.map_out as *const f32, n * 4);
        let mut bands = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for band in 0..4 {
            bands[band] = raw[band * n..(band + 1) * n].to_vec();
        }
        Ok(bands)
    }
}

unsafe fn ensure_dwt_buffers(gpu: &mut Gpu, in_bytes: u64, out_bytes: u64) -> Result<(), String> {
    if gpu.buf_in != 0 && gpu.cap_in >= in_bytes && gpu.cap_out >= out_bytes {
        return Ok(());
    }
    if gpu.buf_in != 0 {
        (gpu.procs.destroy_buffer)(gpu.device, gpu.buf_in, std::ptr::null());
        (gpu.procs.free_mem)(gpu.device, gpu.mem_in, std::ptr::null());
        (gpu.procs.destroy_buffer)(gpu.device, gpu.buf_out, std::ptr::null());
        (gpu.procs.free_mem)(gpu.device, gpu.mem_out, std::ptr::null());
        gpu.buf_in = 0;
    }
    let in_size = in_bytes.max(gpu.cap_in).max(256);
    let out_size = out_bytes.max(gpu.cap_out).max(256);
    let (buf_in, mem_in, map_in) = make_buffer(gpu, in_size, BUFFER_STORAGE, Place::HostShared)?;
    let (buf_out, mem_out, map_out) = match make_buffer(gpu, out_size, BUFFER_STORAGE, Place::HostShared) {
        Ok(buf) => buf,
        Err(err) => {
            (gpu.procs.destroy_buffer)(gpu.device, buf_in, std::ptr::null());
            (gpu.procs.free_mem)(gpu.device, mem_in, std::ptr::null());
            return Err(err);
        }
    };
    let infos = [
        BufDesc { buffer: buf_in, offset: 0, range: WHOLE },
        BufDesc { buffer: buf_out, offset: 0, range: WHOLE },
    ];
    let writes = [
        WriteSet {
            s_type: 35,
            p_next: std::ptr::null(),
            dst: gpu.set,
            binding: 0,
            element: 0,
            count: 1,
            desc_type: DESC_STORAGE,
            image: std::ptr::null(),
            buffer: &infos[0],
            texel: std::ptr::null(),
        },
        WriteSet {
            s_type: 35,
            p_next: std::ptr::null(),
            dst: gpu.set,
            binding: 1,
            element: 0,
            count: 1,
            desc_type: DESC_STORAGE,
            image: std::ptr::null(),
            buffer: &infos[1],
            texel: std::ptr::null(),
        },
    ];
    (gpu.procs.update_sets)(gpu.device, 2, writes.as_ptr(), 0, std::ptr::null());
    gpu.buf_in = buf_in;
    gpu.mem_in = mem_in;
    gpu.map_in = map_in;
    gpu.buf_out = buf_out;
    gpu.mem_out = mem_out;
    gpu.map_out = map_out;
    gpu.cap_in = in_size;
    gpu.cap_out = out_size;
    Ok(())
}

unsafe fn submit_level(gpu: &Gpu, push: &DwtPush, groups_x: u32, groups_y: u32) -> Result<(), String> {
    vk((gpu.procs.reset_pool)(gpu.device, gpu.cmd_pool, 0), "vkResetCommandPool")?;
    let begin = BeginInfo { s_type: 42, p_next: std::ptr::null(), flags: 0x1, inherit: std::ptr::null() };
    vk((gpu.procs.begin)(gpu.cmd, &begin), "vkBeginCommandBuffer")?;
    buf_barrier(gpu, gpu.buf_in, STAGE_HOST, STAGE_COMPUTE, ACCESS_HOST_WRITE, ACCESS_SHADER_READ);
    buf_barrier(gpu, gpu.buf_out, STAGE_HOST, STAGE_COMPUTE, ACCESS_HOST_WRITE, ACCESS_SHADER_WRITE);
    (gpu.procs.bind_pipe)(gpu.cmd, BIND_COMPUTE, gpu.pipeline);
    (gpu.procs.bind_sets)(gpu.cmd, BIND_COMPUTE, gpu.layout, 0, 1, &gpu.set, 0, std::ptr::null());
    (gpu.procs.push)(
        gpu.cmd,
        gpu.layout,
        STAGE_COMPUTE,
        0,
        std::mem::size_of::<DwtPush>() as u32,
        push as *const DwtPush as *const u8,
    );
    (gpu.procs.dispatch)(gpu.cmd, groups_x, groups_y, 1);
    buf_barrier(gpu, gpu.buf_out, STAGE_COMPUTE, STAGE_HOST, ACCESS_SHADER_WRITE, ACCESS_HOST_READ);
    vk((gpu.procs.end)(gpu.cmd), "vkEndCommandBuffer")?;
    vk((gpu.procs.reset_fence)(gpu.device, 1, &gpu.fence), "vkResetFences")?;
    let submit = SubmitInfo {
        s_type: 4,
        p_next: std::ptr::null(),
        wait_count: 0,
        waits: std::ptr::null(),
        wait_stages: std::ptr::null(),
        cmd_count: 1,
        cmds: &gpu.cmd,
        signal_count: 0,
        signals: std::ptr::null(),
    };
    vk((gpu.procs.submit)(gpu.queue, 1, &submit, gpu.fence), "vkQueueSubmit")?;
    vk((gpu.procs.wait)(gpu.device, 1, &gpu.fence, 1, 30_000_000_000), "vkWaitForFences")?;
    Ok(())
}

fn create_device_inner(allow_software: bool) -> Result<Device, String> {
    if SPIRV.len() < 20 || SPIRV.len() % 4 != 0 || SPIRV[0..4] != [0x03, 0x02, 0x23, 0x07] {
        return Err("embedded SPIR-V is not a shader module".into());
    }
    let (lib, get) = open_loader()?;
    unsafe {
        let create_instance: FnCreateInstance = load(&get, 0, "vkCreateInstance")?;
        let app_name = CString::new("pyrowave-rs").unwrap();
        let app = AppInfo {
            s_type: 0,
            p_next: std::ptr::null(),
            name: app_name.as_ptr(),
            app_version: 0,
            engine: std::ptr::null(),
            engine_version: 0,
            api: API_1_0,
        };
        let info = InstanceInfo {
            s_type: 1,
            p_next: std::ptr::null(),
            flags: 0,
            app: &app,
            layer_count: 0,
            layers: std::ptr::null(),
            ext_count: 0,
            exts: std::ptr::null(),
        };
        let mut instance = 0u64;
        vk(create_instance(&info, std::ptr::null(), &mut instance), "vkCreateInstance")?;
        let enum_phys: FnEnumPhys = load(&get, instance, "vkEnumeratePhysicalDevices")?;
        let props: FnProps = load(&get, instance, "vkGetPhysicalDeviceProperties")?;
        let families_fn: FnQueueFamilies = load(&get, instance, "vkGetPhysicalDeviceQueueFamilyProperties")?;
        let create_device: FnCreateDevice = load(&get, instance, "vkCreateDevice")?;
        let get_queue: FnGetQueue = load(&get, instance, "vkGetDeviceQueue")?;
        let mut count = 0u32;
        vk(enum_phys(instance, &mut count, std::ptr::null_mut()), "vkEnumeratePhysicalDevices")?;
        if count == 0 {
            return Err("no physical device".into());
        }
        let mut phys = vec![0u64; count as usize];
        vk(enum_phys(instance, &mut count, phys.as_mut_ptr()), "vkEnumeratePhysicalDevices")?;
        let mut chosen: Option<(u64, u32, String, u32)> = None;
        let mut software_name = String::new();
        for p in phys {
            let mut raw = vec![0u8; 4096];
            props(p, raw.as_mut_ptr());
            let vendor = u32::from_le_bytes(raw[8..12].try_into().unwrap());
            let device_type = u32::from_le_bytes(raw[16..20].try_into().unwrap());
            let name = cstr_at(&raw[20..20 + 256]);
            let software = device_type == 4
                || vendor == 0x10005
                || name_is_software(&name);
            if software {
                software_name = name.clone();
                if !allow_software {
                    continue;
                }
            }
            let mut nfam = 0u32;
            families_fn(p, &mut nfam, std::ptr::null_mut());
            if nfam == 0 {
                continue;
            }
            let mut fams = vec![
                QueueFamily { flags: 0, count: 0, timestamp: 0, gx: 0, gy: 0, gz: 0 };
                nfam as usize
            ];
            families_fn(p, &mut nfam, fams.as_mut_ptr());
            let Some(family) = fams.iter().position(|f| f.flags & QUEUE_COMPUTE != 0) else {
                continue;
            };
            let better = match &chosen {
                None => true,
                Some((_, _, _, old_vendor)) => vendor == 0x10de && *old_vendor != 0x10de,
            };
            if better {
                chosen = Some((p, family as u32, name, vendor));
            }
        }
        let Some((phys, family, name, _vendor)) = chosen else {
            if !software_name.is_empty() && !allow_software {
                return Err(format!("software device '{software_name}' rejected"));
            }
            return Err("no hardware Vulkan device with a compute queue".into());
        };
        let prio = 1.0f32;
        let queue = QueueInfo {
            s_type: 2,
            p_next: std::ptr::null(),
            flags: 0,
            family,
            count: 1,
            priorities: &prio,
        };
        let dev_info = DeviceInfo {
            s_type: 3,
            p_next: std::ptr::null(),
            flags: 0,
            queue_count: 1,
            queues: &queue,
            layer_count: 0,
            layers: std::ptr::null(),
            ext_count: 0,
            exts: std::ptr::null(),
            features: std::ptr::null(),
        };
        let mut device = 0u64;
        vk(create_device(phys, &dev_info, std::ptr::null(), &mut device), "vkCreateDevice")?;
        let mut queue_h = 0u64;
        get_queue(device, family, 0, &mut queue_h);

        let mem_props: FnMemProps = load(&get, instance, "vkGetPhysicalDeviceMemoryProperties")?;
        let mut mem = std::mem::zeroed::<MemProps>();
        mem_props(phys, &mut mem);
        let procs = Procs {
            destroy_instance: load(&get, instance, "vkDestroyInstance")?,
            destroy_device: load(&get, instance, "vkDestroyDevice")?,
            destroy_buffer: load(&get, instance, "vkDestroyBuffer")?,
            free_mem: load(&get, instance, "vkFreeMemory")?,
            create_buffer: load(&get, instance, "vkCreateBuffer")?,
            buf_reqs: load(&get, instance, "vkGetBufferMemoryRequirements")?,
            alloc: load(&get, instance, "vkAllocateMemory")?,
            bind: load(&get, instance, "vkBindBufferMemory")?,
            map: load(&get, instance, "vkMapMemory")?,
            alloc_sets: load(&get, instance, "vkAllocateDescriptorSets")?,
            update_sets: load(&get, instance, "vkUpdateDescriptorSets")?,
            alloc_cmd: load(&get, instance, "vkAllocateCommandBuffers")?,
            begin: load(&get, instance, "vkBeginCommandBuffer")?,
            end: load(&get, instance, "vkEndCommandBuffer")?,
            bind_pipe: load(&get, instance, "vkCmdBindPipeline")?,
            bind_sets: load(&get, instance, "vkCmdBindDescriptorSets")?,
            push: load(&get, instance, "vkCmdPushConstants")?,
            dispatch: load(&get, instance, "vkCmdDispatch")?,
            copy: load(&get, instance, "vkCmdCopyBuffer")?,
            barrier: load(&get, instance, "vkCmdPipelineBarrier")?,
            reset_fence: load(&get, instance, "vkResetFences")?,
            submit: load(&get, instance, "vkQueueSubmit")?,
            wait: load(&get, instance, "vkWaitForFences")?,
            reset_pool: load(&get, instance, "vkResetCommandPool")?,
        };
        let create_shader: FnCreateShader = load(&get, instance, "vkCreateShaderModule")?;
        let shader_info = ShaderInfo {
            s_type: 16,
            p_next: std::ptr::null(),
            flags: 0,
            code_size: SPIRV.len(),
            code: SPIRV.as_ptr() as *const u32,
        };
        let mut shader = 0u64;
        vk(create_shader(device, &shader_info, std::ptr::null(), &mut shader), "vkCreateShaderModule")?;
        let bindings = [
            Binding { binding: 0, desc_type: DESC_STORAGE, count: 1, stages: STAGE_COMPUTE, samplers: std::ptr::null() },
            Binding { binding: 1, desc_type: DESC_STORAGE, count: 1, stages: STAGE_COMPUTE, samplers: std::ptr::null() },
        ];
        let set_info = SetLayoutInfo {
            s_type: 32,
            p_next: std::ptr::null(),
            flags: 0,
            count: 2,
            bindings: bindings.as_ptr(),
        };
        let set_layout_fn: FnSetLayout = load(&get, instance, "vkCreateDescriptorSetLayout")?;
        let mut set_layout = 0u64;
        vk(set_layout_fn(device, &set_info, std::ptr::null(), &mut set_layout), "vkCreateDescriptorSetLayout")?;
        let push = PushRange { stages: STAGE_COMPUTE, offset: 0, size: std::mem::size_of::<Push>() as u32 };
        let pipe_layout_info = PipeLayoutInfo {
            s_type: 30,
            p_next: std::ptr::null(),
            flags: 0,
            set_count: 1,
            sets: &set_layout,
            push_count: 1,
            pushes: &push,
        };
        let pipe_layout_fn: FnPipeLayout = load(&get, instance, "vkCreatePipelineLayout")?;
        let mut layout = 0u64;
        vk(pipe_layout_fn(device, &pipe_layout_info, std::ptr::null(), &mut layout), "vkCreatePipelineLayout")?;
        let entry = CString::new("main").unwrap();
        let stage = StageInfo {
            s_type: 18,
            p_next: std::ptr::null(),
            flags: 0,
            stage: STAGE_COMPUTE,
            module: shader,
            name: entry.as_ptr(),
            spec: std::ptr::null(),
        };
        let compute = ComputeInfo {
            s_type: 29,
            p_next: std::ptr::null(),
            flags: 0,
            stage,
            layout,
            base: 0,
            base_index: -1,
        };
        let pipe_fn: FnComputePipe = load(&get, instance, "vkCreateComputePipelines")?;
        let mut pipeline = 0u64;
        vk(pipe_fn(device, 0, 1, &compute, std::ptr::null(), &mut pipeline), "vkCreateComputePipelines")?;
        let pool_size = PoolSize { desc_type: DESC_STORAGE, count: 2 };
        let pool_info = PoolInfo {
            s_type: 33,
            p_next: std::ptr::null(),
            flags: 0,
            max_sets: 1,
            size_count: 1,
            sizes: &pool_size,
        };
        let pool_fn: FnPool = load(&get, instance, "vkCreateDescriptorPool")?;
        let mut pool = 0u64;
        vk(pool_fn(device, &pool_info, std::ptr::null(), &mut pool), "vkCreateDescriptorPool")?;
        let set_alloc = SetAlloc {
            s_type: 34,
            p_next: std::ptr::null(),
            pool,
            count: 1,
            layouts: &set_layout,
        };
        let mut set = 0u64;
        vk((procs.alloc_sets)(device, &set_alloc, &mut set), "vkAllocateDescriptorSets")?;
        let cmd_pool_info = CmdPoolInfo { s_type: 39, p_next: std::ptr::null(), flags: 0, family };
        let cmd_pool_fn: FnCmdPool = load(&get, instance, "vkCreateCommandPool")?;
        let mut cmd_pool = 0u64;
        vk(cmd_pool_fn(device, &cmd_pool_info, std::ptr::null(), &mut cmd_pool), "vkCreateCommandPool")?;
        let cmd_alloc = CmdAlloc { s_type: 40, p_next: std::ptr::null(), pool: cmd_pool, level: 0, count: 1 };
        let mut cmd = 0u64;
        vk((procs.alloc_cmd)(device, &cmd_alloc, &mut cmd), "vkAllocateCommandBuffers")?;
        let fence_info = FenceInfo { s_type: 8, p_next: std::ptr::null(), flags: 0 };
        let fence_fn: FnFence = load(&get, instance, "vkCreateFence")?;
        let mut fence = 0u64;
        vk(fence_fn(device, &fence_info, std::ptr::null(), &mut fence), "vkCreateFence")?;
        let _ = shader;
        Ok(Device {
            name,
            gpu: std::cell::RefCell::new(Gpu {
                lib,
                instance,
                device,
                queue: queue_h,
                layout,
                pipeline,
                pool,
                set,
                cmd_pool,
                cmd,
                fence,
                family,
                mem,
                buf_in: 0,
                mem_in: 0,
                buf_out: 0,
                mem_out: 0,
                dev_in: 0,
                dev_in_mem: 0,
                dev_out: 0,
                dev_out_mem: 0,
                map_in: std::ptr::null_mut(),
                map_out: std::ptr::null_mut(),
                width: 0,
                height: 0,
                chroma_444: false,
                prefix: 0,
                upload: 0,
                ready: false,
                cap_in: 0,
                cap_out: 0,
                procs,
            }),
        })
    }
}

fn encode_inner(
    device: &Device,
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
    let (cw, ch) = if chroma_444 { (width, height) } else { (width / 2, height / 2) };
    let yb = (width as usize) * (height as usize);
    let cb = (cw as usize) * (ch as usize);
    if y.len() < yb || u.len() < cb || v.len() < cb {
        return Err("plane is short".into());
    }
    let mut gpu = device.gpu.borrow_mut();
    if !gpu.ready || gpu.width != width || gpu.height != height || gpu.chroma_444 != chroma_444 {
        prepare(&mut gpu, width, height, cw, ch, chroma_444)?;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(y.as_ptr(), gpu.map_in, yb);
        std::ptr::copy_nonoverlapping(u.as_ptr(), gpu.map_in.add(yb), cb);
        std::ptr::copy_nonoverlapping(v.as_ptr(), gpu.map_in.add(yb + cb), cb);
        vk((gpu.procs.reset_fence)(gpu.device, 1, &gpu.fence), "vkResetFences")?;
        let submit = SubmitInfo {
            s_type: 4,
            p_next: std::ptr::null(),
            wait_count: 0,
            waits: std::ptr::null(),
            wait_stages: std::ptr::null(),
            cmd_count: 1,
            cmds: &gpu.cmd,
            signal_count: 0,
            signals: std::ptr::null(),
        };
        vk((gpu.procs.submit)(gpu.queue, 1, &submit, gpu.fence), "vkQueueSubmit")?;
        vk((gpu.procs.wait)(gpu.device, 1, &gpu.fence, 1, 2_000_000_000), "vkWaitForFences")?;
        let nbytes = gpu.upload;
        let payload = std::slice::from_raw_parts(gpu.map_out, nbytes);
        let bytes = block_record(0, payload);
        Ok(FrameBits {
            width,
            height,
            chroma_444,
            blocks: vec![BlockMeta { bytes }],
        })
    }
}

fn prepare(gpu: &mut Gpu, width: u32, height: u32, cw: u32, ch: u32, chroma_444: bool) -> Result<(), String> {
    let tiles = tile_total(width, height, cw, ch);
    if tiles == 0 || tiles > 8192 {
        return Err(format!("tile count {tiles} is outside the shader dispatch"));
    }
    let in_bytes = (width as u64) * (height as u64) + 2 * (cw as u64) * (ch as u64);
    let packed_bytes = packed_tile_bytes(tiles);
    let dev_out_bytes = packed_bytes;
    let upload_bytes = in_bytes;
    unsafe {
        if gpu.buf_in != 0 {
            for (buf, mem) in [
                (gpu.buf_in, gpu.mem_in),
                (gpu.buf_out, gpu.mem_out),
                (gpu.dev_in, gpu.dev_in_mem),
                (gpu.dev_out, gpu.dev_out_mem),
            ] {
                if buf != 0 {
                    (gpu.procs.destroy_buffer)(gpu.device, buf, std::ptr::null());
                    (gpu.procs.free_mem)(gpu.device, mem, std::ptr::null());
                }
            }
            gpu.buf_in = 0;
            gpu.dev_in = 0;
        }
        let (buf_in, mem_in, map_in) = make_buffer(gpu, in_bytes, BUFFER_TRANSFER_SRC, Place::HostWrite)?;
        let (dev_in, dev_in_mem, _) = make_buffer(gpu, in_bytes, BUFFER_TRANSFER_DST | BUFFER_STORAGE, Place::Device)?;
        let (dev_out, dev_out_mem, _) = make_buffer(gpu, dev_out_bytes, BUFFER_STORAGE | BUFFER_TRANSFER_SRC, Place::Device)?;
        let (buf_out, mem_out, map_out) = make_buffer(gpu, packed_bytes, BUFFER_TRANSFER_DST, Place::HostRead)?;
        let infos = [
            BufDesc { buffer: dev_in, offset: 0, range: WHOLE },
            BufDesc { buffer: dev_out, offset: 0, range: WHOLE },
        ];
        let writes = [
            WriteSet {
                s_type: 35,
                p_next: std::ptr::null(),
                dst: gpu.set,
                binding: 0,
                element: 0,
                count: 1,
                desc_type: DESC_STORAGE,
                image: std::ptr::null(),
                buffer: &infos[0],
                texel: std::ptr::null(),
            },
            WriteSet {
                s_type: 35,
                p_next: std::ptr::null(),
                dst: gpu.set,
                binding: 1,
                element: 0,
                count: 1,
                desc_type: DESC_STORAGE,
                image: std::ptr::null(),
                buffer: &infos[1],
                texel: std::ptr::null(),
            },
        ];
        (gpu.procs.update_sets)(gpu.device, 2, writes.as_ptr(), 0, std::ptr::null());
        let push = Push {
            width,
            height,
            cw,
            ch,
            prefix: tiles,
            energy: tiles * 512,
        };
        let begin = BeginInfo { s_type: 42, p_next: std::ptr::null(), flags: 0x2, inherit: std::ptr::null() };
        vk((gpu.procs.begin)(gpu.cmd, &begin), "vkBeginCommandBuffer")?;
        buf_barrier(gpu, buf_in, STAGE_HOST, STAGE_TRANSFER, ACCESS_HOST_WRITE, ACCESS_TRANSFER_READ);
        let in_copy = CopyRegion { src: 0, dst: 0, size: upload_bytes };
        (gpu.procs.copy)(gpu.cmd, buf_in, dev_in, 1, &in_copy);
        buf_barrier(gpu, dev_in, STAGE_TRANSFER, STAGE_COMPUTE, ACCESS_TRANSFER_WRITE, ACCESS_SHADER_READ);
        (gpu.procs.bind_pipe)(gpu.cmd, BIND_COMPUTE, gpu.pipeline);
        (gpu.procs.bind_sets)(gpu.cmd, BIND_COMPUTE, gpu.layout, 0, 1, &gpu.set, 0, std::ptr::null());
        (gpu.procs.push)(
            gpu.cmd,
            gpu.layout,
            STAGE_COMPUTE,
            0,
            std::mem::size_of::<Push>() as u32,
            &push as *const Push as *const u8,
        );
        // One group per luma and chroma tile. Each group lifts, quantizes, and packs.
        (gpu.procs.dispatch)(gpu.cmd, tiles, 1, 1);
        buf_barrier(gpu, dev_out, STAGE_COMPUTE, STAGE_TRANSFER, ACCESS_SHADER_WRITE, ACCESS_TRANSFER_READ);
        let out_copy = CopyRegion { src: 0, dst: 0, size: packed_bytes };
        (gpu.procs.copy)(gpu.cmd, dev_out, buf_out, 1, &out_copy);
        buf_barrier(gpu, buf_out, STAGE_TRANSFER, STAGE_HOST, ACCESS_TRANSFER_WRITE, ACCESS_HOST_READ);
        vk((gpu.procs.end)(gpu.cmd), "vkEndCommandBuffer")?;
        gpu.buf_in = buf_in;
        gpu.mem_in = mem_in;
        gpu.buf_out = buf_out;
        gpu.mem_out = mem_out;
        gpu.dev_in = dev_in;
        gpu.dev_in_mem = dev_in_mem;
        gpu.dev_out = dev_out;
        gpu.dev_out_mem = dev_out_mem;
        gpu.map_in = map_in;
        gpu.map_out = map_out;
        gpu.width = width;
        gpu.height = height;
        gpu.chroma_444 = chroma_444;
        gpu.prefix = tiles;
        gpu.upload = packed_bytes as usize;
        gpu.ready = true;
        let _ = gpu.family;
    }
    Ok(())
}

enum Place {
    HostWrite,
    Device,
    HostRead,
    HostShared,
}

unsafe fn make_buffer(gpu: &Gpu, size: u64, usage: u32, place: Place) -> Result<(u64, u64, *mut u8), String> {
    let info = BufferInfo {
        s_type: 12,
        p_next: std::ptr::null(),
        flags: 0,
        size,
        usage,
        sharing: 0,
        family_count: 0,
        families: std::ptr::null(),
    };
    let mut buffer = 0u64;
    vk((gpu.procs.create_buffer)(gpu.device, &info, std::ptr::null(), &mut buffer), "vkCreateBuffer")?;
    let mut req = MemReq { size: 0, alignment: 0, bits: 0, _pad: 0 };
    (gpu.procs.buf_reqs)(gpu.device, buffer, &mut req);
    let index = match mem_index(&gpu.mem, req.bits, &place) {
        Ok(index) => index,
        Err(err) => {
            (gpu.procs.destroy_buffer)(gpu.device, buffer, std::ptr::null());
            return Err(err);
        }
    };
    let alloc = AllocInfo { s_type: 5, p_next: std::ptr::null(), size: req.size.max(size), index };
    let mut memory = 0u64;
    vk((gpu.procs.alloc)(gpu.device, &alloc, std::ptr::null(), &mut memory), "vkAllocateMemory")?;
    vk((gpu.procs.bind)(gpu.device, buffer, memory, 0), "vkBindBufferMemory")?;
    let mut mapped = std::ptr::null_mut();
    if !matches!(place, Place::Device) {
        vk((gpu.procs.map)(gpu.device, memory, 0, WHOLE, 0, &mut mapped), "vkMapMemory")?;
    }
    Ok((buffer, memory, mapped as *mut u8))
}

fn mem_index(mem: &MemProps, bits: u32, place: &Place) -> Result<u32, String> {
    let mut fallback = None;
    for i in 0..mem.type_count {
        if bits & (1 << i) == 0 {
            continue;
        }
        let flags = mem.types[i as usize].flags;
        match place {
            Place::Device => {
                if flags & MEM_DEVICE_LOCAL == 0 {
                    continue;
                }
                if flags & MEM_HOST_VISIBLE == 0 {
                    return Ok(i);
                }
                fallback = Some(i);
            }
            Place::HostRead => {
                let host = MEM_HOST_VISIBLE | MEM_HOST_COHERENT;
                if flags & host != host {
                    continue;
                }
                if flags & MEM_HOST_CACHED != 0 {
                    return Ok(i);
                }
                fallback = Some(i);
            }
            Place::HostShared => {
                let host = MEM_HOST_VISIBLE | MEM_HOST_COHERENT;
                if flags & host != host {
                    continue;
                }
                if flags & MEM_HOST_CACHED != 0 {
                    return Ok(i);
                }
                fallback = Some(i);
            }
            Place::HostWrite => {
                let host = MEM_HOST_VISIBLE | MEM_HOST_COHERENT;
                if flags & host != host {
                    continue;
                }
                if flags & MEM_DEVICE_LOCAL == 0 && flags & MEM_HOST_CACHED == 0 {
                    return Ok(i);
                }
                fallback = Some(i);
            }
        }
    }
    fallback.ok_or_else(|| "no compatible Vulkan memory type".to_string())
}

unsafe fn buf_barrier(gpu: &Gpu, buffer: u64, src_stage: u32, dst_stage: u32, src_access: u32, dst_access: u32) {
    let b = BufBarrier {
        s_type: 44,
        p_next: std::ptr::null(),
        src_access,
        dst_access,
        src_queue: u32::MAX,
        dst_queue: u32::MAX,
        buffer,
        offset: 0,
        size: WHOLE,
    };
    (gpu.procs.barrier)(
        gpu.cmd,
        src_stage,
        dst_stage,
        0,
        0,
        std::ptr::null(),
        1,
        &b,
        0,
        std::ptr::null(),
    );
}

/// 1024 coefficients per tile, 2 bits each.
fn packed_tile_bytes(tiles: u32) -> u64 {
    (tiles as u64) * 256
}

fn tile_total(width: u32, height: u32, cw: u32, ch: u32) -> u32 {
    let tx = |n: u32| (n + 31) / 32;
    tx(width) * tx(height) + 2 * tx(cw) * tx(ch)
}

fn name_is_software(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    n.contains("lavapipe") || n.contains("llvmpipe") || n.contains("swiftshader")
}

fn cstr_at(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|b| *b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

fn vk(res: i32, what: &str) -> Result<(), String> {
    if res == 0 {
        Ok(())
    } else {
        Err(format!("{what} returned {res}"))
    }
}

unsafe fn load<T: Copy>(get: &GetProc, obj: u64, name: &str) -> Result<T, String> {
    let c = CString::new(name).unwrap();
    let p = get(obj, c.as_ptr());
    if p.is_null() {
        return Err(format!("missing {name}"));
    }
    Ok(std::mem::transmute_copy(&p))
}

#[cfg(unix)]
fn open_loader() -> Result<(*mut c_void, GetProc), String> {
    // NixOS keeps the loader and ICDs off the default soname path. Publish the
    // ICD JSON list only when the caller has not already chosen drivers.
    publish_driver_files();
    unsafe {
        for name in [b"libvulkan.so.1\0".as_slice(), b"libvulkan.so\0".as_slice()] {
            if let Some(got) = open_named(name.as_ptr() as *const c_char) {
                return Ok(got);
            }
        }
        let mut paths = nix_store_loaders();
        let run = std::path::PathBuf::from("/run/opengl-driver/lib/libvulkan.so.1");
        if run.is_file() {
            paths.push(run);
        }
        paths.sort_by(|a, b| loader_rank(b).cmp(&loader_rank(a)));
        let mut last = String::from("libvulkan.so.1: dlopen failed");
        for path in &paths {
            let text = path.to_string_lossy();
            let Ok(c) = CString::new(text.as_ref()) else {
                continue;
            };
            if let Some(got) = open_named(c.as_ptr()) {
                return Ok(got);
            }
            last = format!("{text}: {}", dlerror_text());
        }
        if paths.is_empty() {
            let extra = dlerror_text();
            if !extra.is_empty() {
                last = format!("libvulkan.so.1: dlopen failed: {extra}");
            }
        }
        Err(last)
    }
}

#[cfg(unix)]
unsafe fn open_named(name: *const c_char) -> Option<(*mut c_void, GetProc)> {
    let handle = dlopen(name, 2);
    if handle.is_null() {
        return None;
    }
    let sym = dlsym(handle, b"vkGetInstanceProcAddr\0".as_ptr() as *const c_char);
    if sym.is_null() {
        dlclose(handle);
        return None;
    }
    let get: GetProc = std::mem::transmute(sym);
    Some((handle, get))
}

/// Highest `vulkan-loader` version under `/nix/store`, without a store hash.
#[cfg(unix)]
fn nix_store_loaders() -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir("/nix/store") else {
        return out;
    };
    for ent in rd.flatten() {
        let name = ent.file_name();
        let name = name.to_string_lossy();
        if !name.contains("-vulkan-loader-") {
            continue;
        }
        let lib = ent.path().join("lib").join("libvulkan.so.1");
        if lib.is_file() {
            out.push(lib);
        }
    }
    out
}

#[cfg(unix)]
fn loader_rank(path: &std::path::Path) -> (u32, u32, u32, u32) {
    let text = path.to_string_lossy();
    let Some(rest) = text.split("-vulkan-loader-").nth(1) else {
        return (0, 0, 0, 0);
    };
    let ver = rest.split('/').next().unwrap_or("");
    let mut parts = ver.split('.').map(|p| p.parse::<u32>().unwrap_or(0));
    (
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
    )
}

#[cfg(unix)]
fn getenv_set(key: &str) -> bool {
    let Ok(k) = CString::new(key) else {
        return false;
    };
    extern "C" {
        fn getenv(name: *const c_char) -> *const c_char;
    }
    unsafe { !getenv(k.as_ptr()).is_null() }
}

/// Point the loader at one ICD JSON (for example lavapipe) when the caller has
/// not already set `VK_DRIVER_FILES` or `VK_ICD_FILENAMES`.
#[cfg(all(unix, test))]
fn pin_named_icd(needle: &str) {
    if getenv_set("VK_DRIVER_FILES") || getenv_set("VK_ICD_FILENAMES") {
        return;
    }
    let dir = std::path::Path::new("/run/opengl-driver/share/vulkan/icd.d");
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let needle = needle.to_ascii_lowercase();
    let Some(path) = rd.flatten().map(|ent| ent.path()).find(|p| {
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("").to_ascii_lowercase();
        name.contains(&needle) && name.ends_with(".json")
    }) else {
        return;
    };
    let Ok(key) = CString::new("VK_DRIVER_FILES") else {
        return;
    };
    let Ok(val) = CString::new(path.to_string_lossy().as_ref()) else {
        return;
    };
    extern "C" {
        fn setenv(name: *const c_char, value: *const c_char, overwrite: i32) -> i32;
    }
    unsafe {
        let _ = setenv(key.as_ptr(), val.as_ptr(), 0);
    }
}

#[cfg(all(not(unix), test))]
fn pin_named_icd(_needle: &str) {}

#[cfg(unix)]
fn publish_driver_files() {
    if getenv_set("VK_DRIVER_FILES") || getenv_set("VK_ICD_FILENAMES") {
        return;
    }
    let dir = std::path::Path::new("/run/opengl-driver/share/vulkan/icd.d");
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let mut files: Vec<std::path::PathBuf> = rd
        .flatten()
        .map(|ent| ent.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("json"))
        .collect();
    files.sort();
    if files.is_empty() {
        return;
    }
    let joined = files.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(":");
    let Ok(key) = CString::new("VK_DRIVER_FILES") else {
        return;
    };
    let Ok(val) = CString::new(joined) else {
        return;
    };
    extern "C" {
        fn setenv(name: *const c_char, value: *const c_char, overwrite: i32) -> i32;
    }
    unsafe {
        let _ = setenv(key.as_ptr(), val.as_ptr(), 0);
    }
}

#[cfg(unix)]
fn dlerror_text() -> String {
    extern "C" {
        fn dlerror() -> *mut c_char;
    }
    unsafe {
        let p = dlerror();
        if p.is_null() {
            return String::new();
        }
        std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

#[cfg(unix)]
unsafe fn dlopen(name: *const c_char, flags: i32) -> *mut c_void {
    extern "C" {
        fn dlopen(filename: *const c_char, flags: i32) -> *mut c_void;
    }
    dlopen(name, flags)
}

#[cfg(unix)]
unsafe fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void {
    extern "C" {
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }
    dlsym(handle, name)
}

#[cfg(unix)]
unsafe fn dlclose(handle: *mut c_void) {
    extern "C" {
        fn dlclose(handle: *mut c_void) -> i32;
    }
    let _ = dlclose(handle);
}

#[cfg(windows)]
fn open_loader() -> Result<(*mut c_void, GetProc), String> {
    unsafe {
        extern "system" {
            fn LoadLibraryA(name: *const c_char) -> *mut c_void;
            fn LoadLibraryExA(name: *const c_char, file: *mut c_void, flags: u32) -> *mut c_void;
            fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
            fn GetLastError() -> u32;
            fn GetEnvironmentVariableA(name: *const c_char, buf: *mut c_char, n: u32) -> u32;
            fn SetEnvironmentVariableA(name: *const c_char, value: *const c_char) -> i32;
        }
        // LOAD_WITH_ALTERED_SEARCH_PATH: the lavapipe shim loads vulkan_lvp.dll
        // from its own directory. A bare name would not find that sibling.
        const ALTERED_SEARCH: u32 = 0x0000_0008;
        let mut last_err = 0u32;
        let mut handle = LoadLibraryA(b"vulkan-1.dll\0".as_ptr() as *const c_char);
        if handle.is_null() {
            last_err = GetLastError();
            // Lavapipe reads this when the ICD is first loaded. 128 and 256
            // fail encoder_create; 512 is the width the bundled driver was built for.
            let mut width = [0u8; 8];
            let have = GetEnvironmentVariableA(
                b"LP_NATIVE_VECTOR_WIDTH\0".as_ptr() as *const c_char,
                width.as_mut_ptr() as *mut c_char,
                8,
            );
            if have == 0 {
                SetEnvironmentVariableA(
                    b"LP_NATIVE_VECTOR_WIDTH\0".as_ptr() as *const c_char,
                    b"512\0".as_ptr() as *const c_char,
                );
            }
            handle = LoadLibraryExA(
                b"C:\\Program Files\\etherquic\\vulkan-cpu\\vulkan-1.dll\0".as_ptr() as *const c_char,
                std::ptr::null_mut(),
                ALTERED_SEARCH,
            );
            if handle.is_null() {
                last_err = GetLastError();
            }
        }
        if handle.is_null() {
            return Err(format!("vulkan-1.dll: LoadLibrary failed {last_err}"));
        }
        let sym = GetProcAddress(handle, b"vkGetInstanceProcAddr\0".as_ptr() as *const c_char);
        if sym.is_null() {
            return Err("vulkan-1.dll is missing vkGetInstanceProcAddr".into());
        }
        let get: GetProc = std::mem::transmute(sym);
        Ok((handle, get))
    }
}

#[cfg(not(any(unix, windows)))]
fn open_loader() -> Result<(*mut c_void, GetProc), String> {
    Err("no Vulkan loader on this target".into())
}

#[cfg(test)]
mod tests {
    #[test]
    fn spirv_module_is_embedded() {
        let s = super::SPIRV;
        assert!(s.len() > 64 && s.len() % 4 == 0);
        assert_eq!(&s[..4], &[0x03, 0x02, 0x23, 0x07]);
    }

    #[test]
    fn wavelet_level_matches_host() {
        super::pin_named_icd("lvp");
        let dev = match super::create_device(true) {
            Ok(dev) => dev,
            Err(err) => {
                eprintln!("skip wavelet match: {err}");
                return;
            }
        };
        let mut gpu = dev.gpu.borrow_mut();
        let cases = [(64, 64, 128, 128, true), (96, 80, 128, 128, true), (96, 80, 96, 96, false)];
        for (rw, rh, aw, ah, dc) in cases {
            let mut src = vec![0f32; (rw * rh) as usize];
            let mut state = 1u32;
            for p in &mut src {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let byte = (state >> 24) as u8;
                *p = if dc { f32::from(byte) / 255.0 } else { (f32::from(byte) - 128.0) / 255.0 };
            }
            let host = crate::rdo_dwt::forward_level(&src, rw, rh, aw, ah, dc);
            let got = super::dispatch_level(&mut gpu, &src, rw, rh, aw, ah, dc).expect("dispatch");
            let mut diffs = 0u32;
            for b in 0..4 {
                assert_eq!(host[b].len(), got[b].len(), "band {b} len");
                for (i, (a, c)) in host[b].iter().zip(got[b].iter()).enumerate() {
                    if a.to_bits() != c.to_bits() {
                        diffs += 1;
                        if diffs <= 6 {
                            eprintln!(
                                "case {rw}x{rh} dc={dc} band {b} i {i} host {a:e} gpu {c:e} {:08x}/{:08x}",
                                a.to_bits(),
                                c.to_bits()
                            );
                        }
                    }
                }
            }
            assert_eq!(diffs, 0, "case {rw}x{rh} aligned {aw}x{ah} dc={dc}");
        }
    }

    #[test]
    fn tile_count_matches_the_32_grid() {
        let t720 = super::tile_total(1280, 720, 640, 360);
        let t1080 = super::tile_total(1920, 1080, 960, 540);
        assert_eq!(t720, 40 * 23 + 2 * 20 * 12);
        assert_eq!(t1080, 60 * 34 + 2 * 30 * 17);
        assert_eq!(super::packed_tile_bytes(t720), (t720 as u64) * 256);
        assert!(super::packed_tile_bytes(t720) > 60 * 256);
        assert!(super::packed_tile_bytes(t1080) > 96 * 256);
    }
}
