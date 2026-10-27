# pyrowave-rs

CPU (`std::simd`), D3D11 `cs_5_0`, OpenGL 4.3, and Vulkan encode through one Rust packetizer. The GPU backends dispatch only the wavelet. Quant, analyze, resolve, and RDO pack are the shared CPU path in `src/rdo.rs`.

Bit-exact packets across those backends are the constraint. The shipped wavelet is `rdo_dwt::forward_level`: `Simd<f32, 2>` pairs, scalar `f32_to_f16` at the shared store, and a contracted `mul_add`. Do not widen that lift. `simd_dwt`'s `f32x8` wavelet is a separate path and is not what `Encoder` calls.

## Worth doing

1. Reuse LL and band buffers instead of cloning them every level. `build_stored` clones the LL band to feed the next level, and `encode_stored` clones every band into `Band.pixels`. `xform8` and `xform4` allocate a pending list per 32×32 tile. Keep one scratch set on `Encoder` and reuse it across frames.
2. Convert planes and quantize the 8-texel lanes with `f32x8`. `plane_from_u8` divides each byte by 255 on its own. `quant_band` allocates a `Vec` per 8×8 block and walks the eight texels with scalar `abs` and `as i32`. An `f32x8` max and scale keeps the current cluster order.
3. Use f16c for the pair pack and unpack where `half_matches_f16c_rne` already matches. `_mm_cvtps_ph` is the same rounding as `f32_to_f16` on the hosts that have the feature (this NixOS box and AN515).
4. Double-buffer the GPU upload against the CPU pack. Each backend waits for the wavelet readback, then runs the CPU tail. A second command buffer can upload frame N+1 while the CPU packs frame N. That is a double-buffered `Encoder`, not an async runtime. Levels of one component stay serial because each level reads the previous LL. Y, U, and V at one level do not depend on each other.
5. Leave the `f32x2` lift order alone. That order is what keeps the RTX 2060 SUPER, lavapipe, and the Radeon HD 6670 on the same packet bytes.

No third-party crates are required for 1–3. `rayon` is the only crate worth adding, and only if the three components at one level share the scratch from item 1. `analyze_band` advances a shared `consumed` counter, so bands cannot be packed in parallel without changing the bitstream.
