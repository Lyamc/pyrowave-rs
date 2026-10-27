//! Two target budgets on the shipped CPU encoder. Length tracks the target.

use pyrowave_rs::{parse_sequence_header, Backend, Chroma, CreateInfo, Encoder, YuvFrame};

fn plane(w: usize, h: usize, seed: u32) -> Vec<u8> {
    let mut p = vec![0u8; w * h];
    let mut s = seed;
    for y in 0..h {
        for x in 0..w {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let v = 128.0
                + 40.0 * ((x as f32) * 0.07).sin() * ((y as f32) * 0.05).cos()
                + ((s >> 24) as i32 as f32);
            p[y * w + x] = v.clamp(0.0, 255.0) as u8;
        }
    }
    p
}

fn run(target: u32) -> (usize, u32, u32, usize) {
    let w = 64u32;
    let h = 64u32;
    let y = plane(w as usize, h as usize, 1);
    let u = plane(w as usize / 2, h as usize / 2, 2);
    let v = plane(w as usize / 2, h as usize / 2, 3);
    let mut enc = Encoder::create(CreateInfo {
        backend: Backend::Cpu,
        allow_software: false,
    })
    .expect("cpu");
    enc.encode_target(
        &YuvFrame {
            width: w,
            height: h,
            chroma: Chroma::Yuv420,
            y: &y,
            u: &u,
            v: &v,
        },
        target,
    )
    .expect("encode");
    let boundary = 1 << 20;
    let n = enc.compute_num_packets(boundary).expect("count");
    let mut bits = vec![0u8; boundary];
    let packets = enc.packetize(boundary, &mut bits).expect("packetize");
    assert_eq!(packets.len(), n);
    let mut walked = 0usize;
    for packet in packets.iter() {
        assert_eq!(packet.offset, walked, "packet gap or overlap");
        walked += packet.size;
    }
    assert_eq!(walked, packets.iter().map(|p| p.size).sum::<usize>());
    assert!(bits[walked..].iter().all(|b| *b == 0), "bytes past the walked record were written");
    let (hw, hh, chroma_444) = parse_sequence_header(&bits).expect("header");
    assert_eq!((hw, hh, chroma_444), (w, h, false));
    eprintln!("cpu-rdo target={target} bytes={walked} packets={n} header={hw}x{hh} walked={walked}");
    (walked, hw, hh, n)
}

#[test]
fn two_targets_move_the_record() {
    let (hi, _, _, _) = run(1 << 20);
    let (lo, _, _, _) = run(4096);
    assert!(lo < hi, "lower target {lo} is not shorter than {hi}");
    assert_ne!(lo, hi, "two targets collapsed to one size");
}
