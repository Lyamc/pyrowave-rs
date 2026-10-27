# pyrowave-rs

CPU (`std::simd`), D3D11 `cs_5_0`, OpenGL 4.3, and Vulkan encode through one Rust packetizer. The GPU backends dispatch only the wavelet. Quant, analyze, resolve, and RDO pack are the shared CPU path in `src/rdo.rs`.

Bit-exact packets across those backends are the constraint. The shipped wavelet stays `rdo_dwt::forward_level`: `Simd<f32, 2>` pairs, scalar `f32_to_f16` at the shared store, and a contracted `mul_add`. Do not widen that lift. The `simd_dwt` `f32x8` wavelet stays off the `Encoder` path.

Items 1–4 are in. Packet bytes match the shared CPU quant, analyze, resolve, and RDO pack path (`encode_pair` against two `encode_target` calls, lavapipe included). A faster bitstream that differs is a failure.

## Worth doing

1. Done. `Encoder` keeps two `Scratch` slots and reuses their plane and band buffers across frames. The next level reads the previous LL in place. `encode_stored` borrows band pixels. `xform8` / `xform4` clear one thread-local pending list per tile instead of allocating one. No third-party crates. Bands stay serial because `analyze_band` shares a `consumed` counter. Levels of one component stay serial because each level reads the previous LL.
2. Done. `plane_from_u8_into` divides bytes by 255 with `f32x8` (tail stays scalar `/ 255.0`). `quant_band` keeps eight `LaneQ` values on the stack. Each lane's eight texels are scaled with `f32x8`, then truncated with `as i32` in the same part/i order. `cluster_max` is unchanged. No third-party crates.
3. Done for the pair pack. `pack2` uses `_mm_cvtps_ph` with rounding mode 0 when f16c is present, cached once per process. `half_matches_f16c_rne` covers that convert, not `_mm_cvtph_ps`, so `unpack2` stays `f16_to_f32`. The scalar `f32_to_f16` remains the reference for every other convert. No third-party crates.
4. Done. `Encoder::encode_pair` fills the next frame's planes in the other scratch slot, submits that frame's first wavelet level, packs the current frame, then finishes the rest of the pyramid and packs the next frame. Vulkan records that submit on a second command buffer. D3D11 flushes the dispatch and maps the readback after the pack. OpenGL dispatches, then `glFinish` and the readback after the pack. `encode_target` still returns the frame it was given. `clip_raw` uses `encode_pair` for every two frames. Levels of one component stay serial. No async runtime.
5. Do not change the lift. Leave the `f32x2` lift order unchanged. That order is what keeps packet bytes identical on the RTX 2060 SUPER, lavapipe, and the Radeon HD 6670.

