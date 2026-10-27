# NixOS dev shell for pyrowave-rs. Nightly rustc comes from rustup via
# rust-toolchain.toml; this shell only puts the Vulkan loader and libglvnd
# on the runtime path. The encoder prefers an NVIDIA device when that ICD
# is visible. Lavapipe stays available when VK_DRIVER_FILES is unset,
# because the crate publishes /run/opengl-driver/share/vulkan/icd.d then.
#
#   nix-shell
#   cargo test --offline
#   cargo run --release --example clip_raw -- cpu clip.i420 1920 1080 72 24 1 out.raw
{
  pkgs ? import <nixpkgs> { },
}:

let
  libs = with pkgs; [
    vulkan-loader
    libglvnd
    libGL
  ];
in
pkgs.mkShell {
  name = "pyrowave-rs";

  nativeBuildInputs = with pkgs; [
    pkg-config
    vulkan-headers
    vulkan-tools
  ];

  buildInputs = libs;

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath libs;
  LIBRARY_PATH = pkgs.lib.makeLibraryPath libs;

  shellHook = ''
    if [ -d /run/opengl-driver/lib ]; then
      export LD_LIBRARY_PATH="/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH"
    fi
    if [ -z "''${__EGL_VENDOR_LIBRARY_FILENAMES:-}" ] && [ -f /run/opengl-driver/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
      export __EGL_VENDOR_LIBRARY_FILENAMES=/run/opengl-driver/share/glvnd/egl_vendor.d/10_nvidia.json
    fi
    echo "pyrowave-rs nix-shell (vulkan-loader ${pkgs.vulkan-loader.version}, libglvnd ${pkgs.libglvnd.version})"
    echo "  cargo test --offline"
    echo "  cargo run --release --example clip_raw -- cpu <i420> <w> <h> <frames> <fps_num> <fps_den> <out.raw>"
    if [ -z "''${VK_DRIVER_FILES:-}" ]; then
      echo "  VK_DRIVER_FILES is unset; the encoder publishes the NixOS ICD directory and prefers NVIDIA"
      echo "  lavapipe: VK_DRIVER_FILES=/run/opengl-driver/share/vulkan/icd.d/lvp_icd.x86_64.json"
    fi
  '';
}
