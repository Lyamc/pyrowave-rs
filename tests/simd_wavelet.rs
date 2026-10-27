use pyrowave_rs::forward_wavelet;

#[test]
fn simd_wavelet_changes_a_nonconstant_plane() {
    let width = 16;
    let height = 16;
    let mut plane = vec![0.0f32; width * height];
    plane[0] = 10.0;
    plane[3] = 40.0;
    plane[20] = -7.5;
    plane[width * 8 + 9] = 100.0;
    plane[width * 15 + 15] = 3.0;
    let input = plane.clone();
    forward_wavelet(&mut plane, width, height);
    assert!(plane.iter().all(|v| v.is_finite()), "wavelet produced a non-finite sample");
    assert_ne!(plane, input, "wavelet returned the input unchanged");
    eprintln!("simd wavelet {width}x{height} finite and not a copy of the input");
}
