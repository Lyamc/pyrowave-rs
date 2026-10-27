//! PyroWave encoder. One pipeline (band list, wavelet, packetize) with three
//! backends: CPU (`std::simd`), Direct3D 11 feature level 11_0 `cs_5_0`, and
//! OpenGL 4.3 compute.

#![feature(portable_simd)]

mod cpu;
mod d3d11;
mod d3d11_plan;
mod jobs;
mod opengl;
mod packet;
mod rdo;
mod rdo_dwt;
mod rdo_scale;
mod simd_dwt;
mod vulkan;

pub use d3d11::{shader_layout_matches_ports, warp_rejected, FEATURE_LEVEL_11_0, SHADER_TARGET};
pub use d3d11_plan::{
    band_texture_desc, plane_texture_desc, submission_plan, DWT_PUSH_BYTES, FORMAT_R32_FLOAT,
    FORMAT_R8_UNORM, PACK_JOB_BYTES, QUANT_PUSH_BYTES, SRV_DIM_TEXTURE2DARRAY, UAV_DIM_TEXTURE2DARRAY,
};
pub use jobs::{band_jobs, coding_layout, BandJob};
pub use opengl::{software_rejected, GL_MAJOR, GL_MINOR};
pub use packet::{
    parse_sequence_header, payload_words_of, pwv1_header, pwv1_one_frame, write_packet_list, Packet, PacketList,
};
pub use simd_dwt::forward_wavelet;

use packet::{packetize, FrameBits};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    D3d11,
    OpenGl,
    Vulkan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chroma {
    Yuv420,
    Yuv444,
}

#[derive(Clone, Copy, Debug)]
pub struct CreateInfo {
    pub backend: Backend,
    /// Bit that lets D3D11 accept WARP and OpenGL accept llvmpipe, softpipe,
    /// and SwiftShader. Default devices stay on hardware.
    pub allow_software: bool,
}

pub struct YuvFrame<'a> {
    pub width: u32,
    pub height: u32,
    pub chroma: Chroma,
    pub y: &'a [u8],
    pub u: &'a [u8],
    pub v: &'a [u8],
}

enum Gpu {
    None,
    D3d11(d3d11::Device),
    OpenGl(opengl::Device),
    Vulkan(vulkan::Device),
}

pub struct Encoder {
    backend: Backend,
    /// Renderer or adapter name when a GPU device was opened. CPU leaves this empty.
    pub device_name: String,
    frame: Option<FrameBits>,
    gpu: Gpu,
    /// `(sequence_count + 1) & 7` at the start of each encode. The first frame is 1.
    sequence: u32,
}

impl Encoder {
    pub fn create(info: CreateInfo) -> Result<Self, String> {
        let (device_name, gpu) = match info.backend {
            Backend::Cpu => (String::new(), Gpu::None),
            Backend::D3d11 => {
                let device = d3d11::create_device(info.allow_software)?;
                let name = d3d11::device_name(&device).to_string();
                (name, Gpu::D3d11(device))
            }
            Backend::OpenGl => {
                let device = opengl::create_device(info.allow_software)?;
                let name = opengl::device_name(&device).to_string();
                (name, Gpu::OpenGl(device))
            }
            Backend::Vulkan => {
                let device = vulkan::create_device(info.allow_software)?;
                let name = vulkan::device_name(&device).to_string();
                (name, Gpu::Vulkan(device))
            }
        };
        Ok(Self {
            backend: info.backend,
            device_name,
            frame: None,
            gpu,
            sequence: 0,
        })
    }

    /// `W * H * 3 / 16`, at least 32768, rounded down to a multiple of 4.
    pub fn default_target(width: u32, height: u32) -> u32 {
        let b = (width as u64).saturating_mul(height as u64).saturating_mul(3) / 16;
        (b.max(32768) as u32) & !3
    }

    pub fn backend(&self) -> Backend {
        self.backend
    }

    pub fn uses_gpu(&self) -> bool {
        !matches!(self.gpu, Gpu::None)
    }

    /// Encode one 8-bit planar frame at [`Encoder::default_target`].
    pub fn encode(&mut self, frame: &YuvFrame<'_>) -> Result<(), String> {
        self.encode_target(frame, Self::default_target(frame.width, frame.height))
    }

    /// Encode at a caller-chosen payload budget. The budget is the reference
    /// `maximum_bitstream_size`: it includes the 8-byte sequence header, and
    /// resolve sees that size in words after the header is removed.
    /// CPU runs the host wavelet. Vulkan, D3D11, and OpenGL dispatch that
    /// backend's wavelet, then all four share quant, analyze, finalize,
    /// resolve, and RDO pack.
    pub fn encode_target(&mut self, frame: &YuvFrame<'_>, target: u32) -> Result<(), String> {
        let target = target & !3;
        if target == 0 {
            return Err("target payload size is 0".into());
        }
        self.sequence = self.sequence.wrapping_add(1) & 7;
        let sequence = self.sequence;
        let encoded = match &self.gpu {
            Gpu::None => rdo::encode_frame(
                frame.y,
                frame.u,
                frame.v,
                frame.width,
                frame.height,
                frame.chroma,
                target,
                sequence,
            ),
            Gpu::D3d11(device) => d3d11::encode(
                device,
                frame.y,
                frame.u,
                frame.v,
                frame.width,
                frame.height,
                frame.chroma,
                target,
                sequence,
            ),
            Gpu::OpenGl(device) => opengl::encode(
                device,
                frame.y,
                frame.u,
                frame.v,
                frame.width,
                frame.height,
                frame.chroma,
                target,
                sequence,
            ),
            Gpu::Vulkan(device) => vulkan::encode(
                device,
                frame.y,
                frame.u,
                frame.v,
                frame.width,
                frame.height,
                frame.chroma,
                target,
                sequence,
            ),
        }?;
        self.frame = Some(encoded);
        Ok(())
    }

    pub fn compute_num_packets(&self, packet_boundary: usize) -> Result<usize, String> {
        let frame = self.frame.as_ref().ok_or("encode has not been called")?;
        packet::count_packets(frame, packet_boundary)
    }

    /// Write the record into `bitstream` and return one packet per slice.
    /// Packet sizes sum to the number of bytes written.
    pub fn packetize(&self, packet_boundary: usize, bitstream: &mut [u8]) -> Result<packet::PacketList, String> {
        let frame = self.frame.as_ref().ok_or("encode has not been called")?;
        let (packets, nbytes) = packetize(frame, packet_boundary, Some(bitstream))?;
        let sum: usize = packets.iter().map(|p| p.size).sum();
        if sum != nbytes {
            return Err(format!("packet sizes {sum} do not cover the {nbytes}-byte record"));
        }
        Ok(packet::PacketList::from_vec(packets))
    }
}

/// Shipped D3D11 request. Tests read this instead of a copied literal.
pub fn d3d11_feature_level() -> u32 {
    d3d11::FEATURE_LEVEL_11_0
}

pub fn d3d11_shader_target() -> &'static str {
    d3d11::SHADER_TARGET
}

pub fn d3d11_shaders_are_cs5() -> bool {
    d3d11::shader_sources_are_cs5()
}

pub fn opengl_shaders_are_430() -> bool {
    opengl::glsl_is_430()
}

/// Try the real device constructors and return their messages. CPU is absent
/// here on purpose: it does not open a device.
pub fn probe_gpu_devices(allow_software: bool) -> (String, String) {
    let d3d = match d3d11::create_device(allow_software) {
        Ok(device) => format!("D3D11 ok: {}", d3d11::device_name(&device)),
        Err(err) => err,
    };
    let gl = match opengl::create_device(allow_software) {
        Ok(device) => format!("OpenGL ok: {}", opengl::device_name(&device)),
        Err(err) => err,
    };
    (d3d, gl)
}
